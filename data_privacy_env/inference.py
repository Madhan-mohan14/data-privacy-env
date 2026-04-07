"""
Data Privacy Env — Baseline Inference Script.

Runs a PII redaction agent against all 3 tasks using the OpenEnv EnvClient.
Follows the official OpenEnv sample inference script pattern.

Usage:
    python inference.py

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     Docker image name (optional, defaults to dataprivacy-env:latest).
"""

import asyncio
import os
import sys
from typing import List

from openai import OpenAI

# Ensure local modules (client.py, models.py) are importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import DataPrivacyEnv  # noqa: E402
from models import DataPrivacyAction  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — GAP 2
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
API_KEY: str = os.getenv("HF_TOKEN", "dummy")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
IMAGE_NAME: str = os.getenv("IMAGE_NAME", "dataprivacy-env:latest")

BENCHMARK: str = "dataprivacy-env"
TASK_NAME: str = "task1_plaintext"          # default / single-task constant
MAX_TOTAL_REWARD: float = 1.6               # fallback: 8 PII items × 0.2 each
MAX_STEPS: int = 20
MAX_TOKENS: int = 150
SUCCESS_SCORE_THRESHOLD: float = 0.6

TASKS: List[str] = ["task1_plaintext", "task2_csv", "task3_json"]

# Per-task maximum achievable reward (used for score normalisation).
# All tasks: 8 PII items × 0.2 = 1.6
TASK_MAX_REWARDS: dict = {
    "task1_plaintext": 1.6,
    "task2_csv": 1.6,
    "task3_json": 1.6,
}


# ---------------------------------------------------------------------------
# Logging helpers — exact format required by the validator
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error=None
) -> None:
    err_str = f" error={error}" if error else " error=null"
    print(
        f"[STEP] step={step} action={action!r} reward={reward:.2f} done={done}{err_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Agent — builds a prompt from the current observation and calls the LLM
# ---------------------------------------------------------------------------
def get_model_message(
    client: OpenAI,
    step: int,
    result,
    last_reward: float,
    history: List[str],
) -> str:
    obs = result.observation
    prompt = f"""You are a PII redaction compliance agent.

TASK: {obs.task_description}
FILES AVAILABLE: {obs.files_in_scope}
LAST RESULT: {obs.last_action_result}
LAST REWARD: {last_reward}
STEP: {obs.step_number} / {obs.max_steps}
CUMULATIVE REWARD: {obs.cumulative_reward}

HISTORY (last 5):
{chr(10).join(history[-5:])}

INSTRUCTIONS:
1. First use list_files to see available files
2. Use read_file to inspect each file
3. Use redact_text to redact EXACT PII strings (copy the exact text)
4. When all PII is redacted, call submit

Respond with ONLY valid JSON — one of:
{{"tool": "list_files", "directory": "."}}
{{"tool": "read_file", "file_path": "<filename>"}}
{{"tool": "redact_text", "file_path": "<file>", "target_string": "<exact PII>", "replacement": "[REDACTED]"}}
{{"tool": "submit"}}"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else '{"tool": "submit"}'
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"tool": "submit"}'


# ---------------------------------------------------------------------------
# Main — GAP 1: uses EnvClient.from_docker_image, not raw httpx
# ---------------------------------------------------------------------------
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await DataPrivacyEnv.from_docker_image(IMAGE_NAME)

    all_scores: List[float] = []

    try:
        for task_id in TASKS:
            print(f"\n{'=' * 50}", flush=True)
            print(f"Running {task_id}", flush=True)
            print(f"{'=' * 50}", flush=True)

            history: List[str] = []
            rewards: List[float] = []
            steps_taken: int = 0
            score: float = 0.0
            success: bool = False

            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

            try:
                result = await env.reset(task_id=task_id)
                last_reward: float = 0.0

                for step in range(1, MAX_STEPS + 1):
                    if result.done:
                        break

                    message = get_model_message(
                        client, step, result, last_reward, history
                    )
                    result = await env.step(DataPrivacyAction(message=message))

                    reward = result.reward or 0.0
                    done = result.done

                    rewards.append(reward)
                    steps_taken = step
                    last_reward = reward

                    log_step(
                        step=step, action=message, reward=reward, done=done, error=None
                    )

                    history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

                    if done:
                        break

                # Per-task score normalisation so every task can reach 1.0
                max_reward = TASK_MAX_REWARDS.get(task_id, MAX_TOTAL_REWARD)
                score = sum(rewards) / max_reward if max_reward > 0 else 0.0
                score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
                success = score >= SUCCESS_SCORE_THRESHOLD

            finally:
                log_end(
                    success=success, steps=steps_taken, score=score, rewards=rewards
                )

            all_scores.append(score)
            print(f"Score for {task_id}: {score:.2f}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\nAverage score across all tasks: {avg:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
