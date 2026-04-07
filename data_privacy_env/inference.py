"""
Data Privacy Env — Inference Script.

Runs a PII redaction agent against all 3 tasks using direct HTTP calls
to the environment server (HF Space or local Docker container).

Usage:
    python inference.py

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    ENV_URL        The environment server URL (default: HF Space URL).
"""

import asyncio
import json as _json
import os
import re
from typing import Dict, List

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_URL: str = os.getenv("ENV_URL", "https://maddy140605-dataprivacy-env.hf.space")

BENCHMARK: str = "dataprivacy-env"
MAX_STEPS: int = 20
MAX_TOKENS: int = 300
SUCCESS_SCORE_THRESHOLD: float = 0.6

TASKS: List[str] = ["task1_plaintext", "task2_csv", "task3_json"]

TASK_MAX_REWARDS: dict = {
    "task1_plaintext": 2.0,
    "task2_csv": 2.0,
    "task3_json": 2.0,
}


# ---------------------------------------------------------------------------
# HTTP helpers — direct calls to the environment server
# ---------------------------------------------------------------------------
async def reset_env(http: httpx.AsyncClient, task_id: str) -> dict:
    resp = await http.post("/reset", json={"task_id": task_id})
    resp.raise_for_status()
    return resp.json()


async def step_env(http: httpx.AsyncClient, message: str) -> dict:
    resp = await http.post("/step", json={"action": {"message": message}})
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Logging helpers — exact format required by the validator
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# JSON extraction — finds the first valid JSON object in model output
# ---------------------------------------------------------------------------
def extract_first_json(text: str) -> str:
    """Extract the first balanced {...} block that parses as valid JSON."""
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start != -1:
                candidate = text[start:i + 1]
                try:
                    _json.loads(candidate)
                    return candidate
                except _json.JSONDecodeError:
                    start = -1
                    depth = 0
    return '{"tool": "submit"}'


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------
def already_redacted(history: List[str]) -> List[str]:
    """Extract target_strings already redacted from history."""
    found = []
    for h in history:
        m = re.search(r'"target_string":\s*"([^"]+)"', h)
        if m:
            found.append(m.group(1))
    return found


# ---------------------------------------------------------------------------
# Agent — builds a prompt and calls the LLM
# ---------------------------------------------------------------------------
def get_model_message(
    client: OpenAI,
    obs: dict,
    last_reward: float,
    history: List[str],
    task_id: str,
    file_cache: Dict[str, str],
) -> str:
    redacted = already_redacted(history)
    redacted_note = (
        f"\nALREADY REDACTED (do NOT redact these again): {redacted}"
        if redacted else ""
    )

    # Inject cached file contents so the model never needs to re-read
    cached_section = ""
    if file_cache:
        cached_section = "\nFILE CONTENTS (already read — do NOT call read_file again for these):\n"
        for fname, content in file_cache.items():
            cached_section += f"=== {fname} ===\n{content}\n"

    # Task-specific workflow instructions
    if task_id == "task3_json":
        workflow = """STEP-BY-STEP WORKFLOW:
1. If breach_report.txt is NOT in FILE CONTENTS → read_file breach_report.txt
2. If medical_database.json is NOT in FILE CONTENTS → read_file medical_database.json
3. Once both files are in FILE CONTENTS:
   - Find the compromised Patient ID from breach_report.txt
   - Find that patient's record in medical_database.json
   - Call redact_text once per sensitive field using the EXACT string value from the JSON:
     fields: name, diagnosis, medication, ssn, phone, email, insurance_id, date_of_birth
4. After redacting all 8 fields → {"tool": "submit"}

CRITICAL RULES:
- Output exactly ONE JSON object per response
- Once a file appears in FILE CONTENTS, NEVER call read_file for it again
- Use the exact field value as target_string (copy character-for-character from the file)"""

    elif task_id == "task2_csv":
        workflow = """STEP-BY-STEP WORKFLOW:
1. If customers.csv is NOT in FILE CONTENTS → read_file customers.csv
2. Once customers.csv is in FILE CONTENTS, redact ALL 8 items in this exact order:
   NAMES (4 items — one per customer row, exact full name string):
   - redact_text for customer 1's name
   - redact_text for customer 2's name
   - redact_text for customer 3's name
   - redact_text for customer 4's name
   SSNs (4 items — format "XXX-XX-XXXX", one per customer row):
   - redact_text for customer 1's SSN
   - redact_text for customer 2's SSN
   - redact_text for customer 3's SSN
   - redact_text for customer 4's SSN
3. Count ALREADY REDACTED items — only submit when count = 8 (4 names + 4 SSNs)
4. Do NOT redact emails or customer IDs

CRITICAL RULES:
- Output exactly ONE JSON object per response
- Once the file appears in FILE CONTENTS, NEVER call read_file again
- Use exact values from the file as target_string
- Do NOT submit until all 8 items (4 names + 4 SSNs) appear in ALREADY REDACTED list"""

    else:  # task1_plaintext
        workflow = """STEP-BY-STEP WORKFLOW:
1. If server_logs.txt is NOT in FILE CONTENTS → read_file server_logs.txt
2. Once server_logs.txt is in FILE CONTENTS:
   - Call redact_text for each email address (exact string from file)
   - Call redact_text for each phone number (exact string from file)
3. After all emails and phone numbers are redacted → {"tool": "submit"}

CRITICAL RULES:
- Output exactly ONE JSON object per response
- Once the file appears in FILE CONTENTS, NEVER call read_file again
- Use exact values from the file as target_string"""

    prompt = f"""You are a PII redaction compliance agent.
Output ONLY a single JSON object — no explanation, no markdown, no extra text before or after.

TASK: {obs.get("task_description", "")}
STEP: {obs.get("step_number", 0)} / {obs.get("max_steps", 20)}
LAST RESULT: {obs.get("last_action_result", "")}
LAST REWARD: {last_reward}
CUMULATIVE REWARD: {obs.get("cumulative_reward", 0.0)}
{redacted_note}
{cached_section}
RECENT HISTORY:
{chr(10).join(history[-5:])}

{workflow}

Valid JSON formats:
{{"tool": "read_file", "file_path": "<filename>"}}
{{"tool": "redact_text", "file_path": "<file>", "target_string": "<exact value>", "replacement": "[REDACTED]"}}
{{"tool": "submit"}}"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        return extract_first_json(raw)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"tool": "submit"}'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_scores: List[float] = []

    async with httpx.AsyncClient(base_url=ENV_URL, timeout=60.0) as http:
        for task_id in TASKS:
            print(f"\n{'=' * 50}", flush=True)
            print(f"Running {task_id}", flush=True)
            print(f"{'=' * 50}", flush=True)

            history: List[str] = []
            rewards: List[float] = []
            file_cache: Dict[str, str] = {}
            steps_taken: int = 0
            score: float = 0.0
            success: bool = False

            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

            try:
                data = await reset_env(http, task_id)
                obs = data.get("observation", {})
                done = data.get("done", False)
                last_reward: float = 0.0

                for step in range(1, MAX_STEPS + 1):
                    if done:
                        break

                    message = get_model_message(
                        llm_client, obs, last_reward, history, task_id, file_cache
                    )

                    try:
                        data = await step_env(http, message)
                    except Exception as exc:
                        print(f"[DEBUG] Step request failed: {exc}", flush=True)
                        break

                    obs = data.get("observation", {})
                    reward = float(data.get("reward") or 0.0)
                    done = bool(data.get("done", False))

                    # Cache file content after a successful read_file
                    try:
                        action_data = _json.loads(message)
                        if action_data.get("tool") == "read_file" and reward >= 0:
                            fname = action_data.get("file_path", "")
                            if fname and fname not in file_cache:
                                file_cache[fname] = str(obs.get("last_action_result", ""))
                    except _json.JSONDecodeError:
                        pass

                    rewards.append(reward)
                    steps_taken = step
                    last_reward = reward

                    log_step(step=step, action=message, reward=reward, done=done)
                    history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

                    if done:
                        break

                max_reward = TASK_MAX_REWARDS.get(task_id, 1.6)
                score = sum(rewards) / max_reward if max_reward > 0 else 0.0
                score = min(max(score, 0.0), 1.0)
                success = score >= SUCCESS_SCORE_THRESHOLD

            finally:
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

            all_scores.append(score)
            print(f"Score for {task_id}: {score:.2f}", flush=True)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\nAverage score across all tasks: {avg:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
