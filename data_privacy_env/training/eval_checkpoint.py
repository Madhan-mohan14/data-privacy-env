"""
eval_checkpoint.py — post-training evaluation for ComplianceGuard.

Runs 30 seeded L3 episodes against a trained checkpoint and saves results
to eval_checkpoint_results.json. Use the output to compare against
baseline_results_seeded.json for the before/after pitch.

Usage:
    python training/eval_checkpoint.py --checkpoint checkpoints/grpo/final-merged
    python training/eval_checkpoint.py --checkpoint checkpoints/grpo/final-merged --level 1
    python training/eval_checkpoint.py --checkpoint checkpoints/grpo/final-merged --episodes 10
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DataPrivacyAction
from training.grpo_env import ComplianceGuardEnvTRL

MAX_STEPS = 30
MAX_NEW_TOKENS = 256


def _load_model(checkpoint: str):
    """Load trained checkpoint. Uses Unsloth if available, falls back to transformers."""
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            checkpoint,
            max_seq_length=2048,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)
        print(f"[Unsloth] Loaded from {checkpoint}")
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print(f"[transformers] Loaded from {checkpoint}")
    return model, tokenizer


def _call_model(model, tokenizer, phase: str, obs_text: str, step: int) -> str:
    """Generate next action JSON from model."""
    from agents.prompts import PHASE_PROMPTS

    phase_key = phase if phase in PHASE_PROMPTS else "SCAN"
    prompt = f"{PHASE_PROMPTS[phase_key]}\n\nStep {step}/{MAX_STEPS}\nLast result: {obs_text[:300]}\nOutput ONE JSON object:"

    try:
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer)
        out = pipe(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)[0]["generated_text"]
        generated = out[len(prompt):]
    except Exception:
        import re
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    import re
    m = re.search(r"\{[^{}]+\}", generated)
    return m.group(0) if m else '{"tool": "submit"}'


def run_episode(model, tokenizer, level: int, seed: int, episode_idx: int) -> dict:
    env = ComplianceGuardEnvTRL()
    env.reset(seed=seed, level=level)
    phase = env.phase
    done = False
    steps = 0

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        phase = env.phase
        action_json = _call_model(model, tokenizer, phase, env.phase, step)

        try:
            obs = env.step(DataPrivacyAction(message=action_json))
        except Exception as e:
            print(f"  step() error: {e}")
            break

        done = obs.done
        steps = step

        if done:
            break

    env.close()
    reward = env.reward
    metrics = {}
    if hasattr(env, '_last_metrics'):
        metrics = env._last_metrics

    # Compute final metrics for reporting even if episode timed out
    if not done:
        reward, metrics = env._compute_reward()

    print(
        f"  ep={episode_idx} seed={seed} level={level} steps={steps} "
        f"reward={reward:.4f} done={done}"
    )
    return {
        "episode": episode_idx,
        "seed": seed,
        "reward": round(reward, 4),
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "steps": steps,
        "done": done,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained ComplianceGuard checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint directory")
    parser.add_argument("--level", type=int, default=3, help="Curriculum level to evaluate (default: 3)")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes (default: 30)")
    parser.add_argument("--out", default="eval_checkpoint_results.json", help="Output JSON path")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer = _load_model(args.checkpoint)

    model_name = os.path.basename(args.checkpoint.rstrip("/\\"))
    results = []

    print(f"\nRunning {args.episodes} episodes on Level {args.level} (seeded)...")
    for i in range(args.episodes):
        result = run_episode(model, tokenizer, level=args.level, seed=i, episode_idx=i)
        results.append(result)

    rewards = [r["reward"] for r in results]
    avg_reward = round(sum(rewards) / len(rewards), 4) if rewards else 0.0
    success_rate = round(sum(1 for r in rewards if r >= 0.7) / len(rewards), 4) if rewards else 0.0

    def _avg_metric(key: str) -> float:
        vals = [r["metrics"].get(key, 0.0) for r in results if r["metrics"]]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {
        "avg_reward": avg_reward,
        "success_rate": success_rate,
        "avg_scan_recall": _avg_metric("scan_recall"),
        "avg_precision": _avg_metric("precision"),
        "avg_redact_completeness": _avg_metric("redact_completeness"),
    }

    output = {
        "checkpoint": args.checkpoint,
        "model": model_name,
        "level": args.level,
        "episodes": results,
        "summary": summary,
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    baseline_l3_avg = 0.000  # from baseline_results_seeded.json
    if os.path.exists("baseline_results_seeded.json"):
        with open("baseline_results_seeded.json") as f:
            b = json.load(f)
            l3_eps = [r for r in b.get("results", []) if r.get("level") == 3]
            if l3_eps:
                baseline_l3_avg = round(sum(r["reward"] for r in l3_eps) / len(l3_eps), 4)

    delta = round(avg_reward - baseline_l3_avg, 4)
    sign = "+" if delta >= 0 else ""
    print(f"\n{'='*60}")
    print(f"Baseline L3 avg: {baseline_l3_avg:.3f} | Checkpoint L3 avg: {avg_reward:.3f} | Delta: {sign}{delta:.3f}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Results saved: {args.out}")


if __name__ == "__main__":
    main()
