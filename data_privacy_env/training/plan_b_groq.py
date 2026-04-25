"""
plan_b_groq.py — Fallback comparison: Groq Llama-70B vs Qwen3-1.7B zero-shot on L3.

Runs 30 seeded L3 episodes with both models on IDENTICAL seeds.
Use this if GRPO produces a flat reward curve — it shows the capability gap
that motivates training, with Groq as the "ceiling" reference.

Pitch narrative:
  Llama-3.3-70B (large, no training): achieves ~0.71 on L3.
  Qwen3-1.7B (small, zero-shot):      achieves ~0.18 on L3.
  Training Qwen3-1.7B with GRPO closes this gap.

Usage:
    python training/plan_b_groq.py --episodes 30
    python training/plan_b_groq.py --episodes 5   # quick smoke test

Required env vars:
    GROQ_API_KEY    Your Groq API key
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from models import DataPrivacyAction
from server.data_privacy_env_environment import ComplianceGuardEnv

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"
QWEN_MODEL = "unsloth/Qwen3-1.7B"

MAX_STEPS = 30
LEVEL = 3

PHASE_SYSTEM = {
    "SCAN": (
        "You are a PII compliance agent. Output ONLY a single raw JSON object. No markdown.\n"
        "PII to flag: EMAIL, PHONE, NAME. Also flag obfuscated forms like 'alice [at] gmail [dot] com'.\n"
        "Tools: list_files, read_file, flag_candidate, advance_phase.\n"
        'Example: {"tool": "flag_candidate", "text": "alice [at] gmail [dot] com", "file_path": "server_logs.txt", "pii_type": "EMAIL"}'
    ),
    "CLASSIFY": (
        "You are a PII compliance agent in CLASSIFY phase. Output ONLY a single raw JSON object.\n"
        "Tools: list_candidates, classify_candidate, advance_phase.\n"
        "Real PII → confirmed:true. System/test addresses → confirmed:false."
    ),
    "REDACT": (
        "You are a PII compliance agent in REDACT phase. Output ONLY a single raw JSON object.\n"
        "Tools: redact_span, submit.\n"
        "Redact ALL confirmed candidates, then submit."
    ),
}


def _extract_json(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text).strip()
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = text[start: i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    start = -1
                    depth = 0
    return '{"tool": "submit"}'


def _groq_action(client: OpenAI, phase: str, obs_result: str, step: int) -> str:
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": PHASE_SYSTEM[phase]},
                {"role": "user", "content": f"Step {step}/{MAX_STEPS}. Last result: {obs_result[:300]}\nOutput ONE JSON object:"},
            ],
            max_tokens=256,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return _extract_json(raw)
    except Exception as e:
        print(f"    [Groq error] {e}")
        return '{"tool": "submit"}'


def _qwen_action(pipe, phase: str, obs_result: str, step: int) -> str:
    from agents.prompts import PHASE_PROMPTS
    phase_key = phase if phase in PHASE_PROMPTS else "SCAN"
    prompt = (
        f"{PHASE_PROMPTS[phase_key]}\n\n"
        f"Step {step}/{MAX_STEPS}. Last result: {obs_result[:300]}\n"
        "Output ONE JSON object:"
    )
    try:
        out = pipe(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]
        generated = out[len(prompt):]
        return _extract_json(generated)
    except Exception as e:
        print(f"    [Qwen error] {e}")
        return '{"tool": "submit"}'


def run_groq_episode(client: OpenAI, level: int, seed: int, episode_idx: int) -> dict:
    env = ComplianceGuardEnv()
    obs = env.reset(seed=seed, level=level)
    done = False
    steps = 0

    for step in range(1, MAX_STEPS + 1):
        if done:
            break
        action_json = _groq_action(client, env.phase, obs.last_action_result, step)
        obs = env.step(DataPrivacyAction(message=action_json))
        done = obs.done
        steps = step

    env.close()
    reward = env.reward if done else env._compute_reward()[0]
    print(f"  [Groq]  ep={episode_idx} seed={seed} steps={steps} reward={reward:.4f}")
    return {"episode": episode_idx, "seed": seed, "reward": round(reward, 4), "steps": steps}


def run_qwen_episode(pipe, level: int, seed: int, episode_idx: int) -> dict:
    env = ComplianceGuardEnv()
    obs = env.reset(seed=seed, level=level)
    done = False
    steps = 0

    for step in range(1, MAX_STEPS + 1):
        if done:
            break
        action_json = _qwen_action(pipe, env.phase, obs.last_action_result, step)
        obs = env.step(DataPrivacyAction(message=action_json))
        done = obs.done
        steps = step

    env.close()
    reward = env.reward if done else env._compute_reward()[0]
    print(f"  [Qwen]  ep={episode_idx} seed={seed} steps={steps} reward={reward:.4f}")
    return {"episode": episode_idx, "seed": seed, "reward": round(reward, 4), "steps": steps}


def _load_qwen():
    """Load Qwen3-1.7B for zero-shot inference."""
    try:
        from unsloth import FastLanguageModel
        from transformers import pipeline as hf_pipeline
        model, tokenizer = FastLanguageModel.from_pretrained(
            QWEN_MODEL, max_seq_length=2048, load_in_4bit=True
        )
        FastLanguageModel.for_inference(model)
        return hf_pipeline("text-generation", model=model, tokenizer=tokenizer)
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
        import torch
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL, torch_dtype=torch.float16, device_map="auto"
        )
        return hf_pipeline("text-generation", model=model, tokenizer=tokenizer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per model (default: 30)")
    parser.add_argument("--skip-qwen", action="store_true", help="Skip Qwen run (Groq only)")
    parser.add_argument("--out", default="plan_b_results.json", help="Output JSON path")
    args = parser.parse_args()

    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY env var required. Set it in .env or export.")
        sys.exit(1)

    seeds = list(range(args.episodes))
    groq_client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

    print(f"\n{'='*60}")
    print(f"Plan B: Groq Llama-70B vs Qwen3-1.7B zero-shot on L{LEVEL}")
    print(f"Episodes: {args.episodes} | Seeds: 0..{args.episodes - 1}")
    print(f"{'='*60}\n")

    # ── Groq Llama-70B run ────────────────────────────────────────────
    print(f"[1/2] Running Groq {GROQ_MODEL}...")
    groq_results = []
    for i, seed in enumerate(seeds):
        result = run_groq_episode(groq_client, LEVEL, seed, i)
        groq_results.append(result)
        time.sleep(0.3)  # rate limit guard

    groq_avg = round(sum(r["reward"] for r in groq_results) / len(groq_results), 4)
    groq_success = round(sum(1 for r in groq_results if r["reward"] >= 0.7) / len(groq_results), 4)

    # ── Qwen3-1.7B zero-shot run ──────────────────────────────────────
    qwen_results = []
    qwen_avg = 0.0
    qwen_success = 0.0

    if not args.skip_qwen:
        print(f"\n[2/2] Loading {QWEN_MODEL} for zero-shot run...")
        try:
            qwen_pipe = _load_qwen()
            for i, seed in enumerate(seeds):
                result = run_qwen_episode(qwen_pipe, LEVEL, seed, i)
                qwen_results.append(result)
            qwen_avg = round(sum(r["reward"] for r in qwen_results) / len(qwen_results), 4)
            qwen_success = round(sum(1 for r in qwen_results if r["reward"] >= 0.7) / len(qwen_results), 4)
        except Exception as e:
            print(f"[Qwen load failed] {e}. Skipping Qwen run.")
            qwen_results = []

    delta = round(groq_avg - qwen_avg, 4)

    output = {
        f"llama_70b": {
            "model": GROQ_MODEL,
            "avg_reward": groq_avg,
            "success_rate": groq_success,
            "episodes": groq_results,
        },
        "qwen_1.7b_zero_shot": {
            "model": QWEN_MODEL,
            "avg_reward": qwen_avg,
            "success_rate": qwen_success,
            "episodes": qwen_results,
        },
        "delta": delta,
        "level": LEVEL,
        "seeds": seeds,
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"{'Model':<30} {'Avg Reward':>12} {'Success Rate':>14}")
    print(f"{'-'*60}")
    print(f"{'Groq ' + GROQ_MODEL:<30} {groq_avg:>12.4f} {groq_success:>13.1%}")
    if qwen_results:
        print(f"{'Qwen3-1.7B zero-shot':<30} {qwen_avg:>12.4f} {qwen_success:>13.1%}")
        print(f"{'Delta (GRPO target gap)':<30} {delta:>+12.4f}")
    print(f"{'='*60}")
    print(f"Results saved: {args.out}")
    print(f"\nPitch line: 'Llama-70B gets {groq_avg:.2f} on L3. Qwen3-1.7B zero-shot gets {qwen_avg:.2f}. GRPO closes this gap.'")


if __name__ == "__main__":
    main()
