"""
ComplianceGuard Inference Script — 3-phase SCAN → CLASSIFY → REDACT.

Runs the env in-process (same pattern as TRL training) — avoids the
stateless HTTP endpoints which create/destroy a fresh env per request.

Usage:
    python inference.py                    # run L1+L3 baseline (30 episodes)
    python inference.py --level 1          # single level
    python inference.py --episodes 5       # quick smoke test

Required env vars (one of):
    HF_TOKEN        HuggingFace token (uses api-inference.huggingface.co + Qwen3)
    GROQ_API_KEY    Groq API key (fallback)
"""

import json as _json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI  # Groq uses OpenAI-compatible API

try:
    from server.data_privacy_env_environment import ComplianceGuardEnv
    from models import DataPrivacyAction
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server.data_privacy_env_environment import ComplianceGuardEnv
    from models import DataPrivacyAction

# ---------------------------------------------------------------------------
# Config — supports both Groq and HF router automatically
# ---------------------------------------------------------------------------
_GROQ_KEY: str = os.getenv("GROQ_API_KEY", "")
_HF_TOKEN: str = os.getenv("HF_TOKEN", "")

if _GROQ_KEY:
    API_KEY: str = _GROQ_KEY
    BASE_URL: str = "https://api.groq.com/openai/v1"
    MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
else:
    API_KEY = _HF_TOKEN
    BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

MAX_STEPS: int = 30
MAX_TOKENS: int = 512

PHASE_TOOLS: dict[str, list[str]] = {
    "SCAN": ["list_files", "read_file", "flag_candidate", "advance_phase"],
    "CLASSIFY": ["list_candidates", "classify_candidate", "advance_phase"],
    "REDACT": ["redact_span", "submit"],
}

PHASE_SYSTEM: dict[str, str] = {
    "SCAN": (
        'You are a PII compliance agent. Output ONLY a single raw JSON object. No markdown.\n'
        'Use "tool" as the key.\n\n'
        'PII to flag (ALL of these):\n'
        '  - EMAIL addresses: "john@email.com" → pii_type: "EMAIL"\n'
        '  - PHONE numbers: "555-123-4567" → pii_type: "PHONE"\n'
        '  - FULL PERSON NAMES like "Grace Harris", "John Smith" → pii_type: "NAME"\n'
        '  - SSNs like "123-45-6789" → pii_type: "SSN"\n\n'
        'WORKFLOW:\n'
        '1. {"tool": "list_files"}  ← only if you don\'t have the file list yet\n'
        '2. {"tool": "read_file", "file_path": "server_logs.txt"}  ← read EACH file\n'
        '3. Flag EVERY PII item found — emails, phones, AND person names:\n'
        '   {"tool": "flag_candidate", "text": "Grace Harris", "file_path": "server_logs.txt", "pii_type": "NAME"}\n'
        '   {"tool": "flag_candidate", "text": "grace.harris@gmail.com", "file_path": "server_logs.txt", "pii_type": "EMAIL"}\n'
        '   {"tool": "flag_candidate", "text": "555-123-4567", "file_path": "server_logs.txt", "pii_type": "PHONE"}\n'
        '4. {"tool": "advance_phase"}  ← ONLY after flagging ALL PII\n\n'
        'CRITICAL: Flag person names too, not just emails and phones!'
    ),
    "CLASSIFY": (
        'You are a PII compliance agent in CLASSIFY phase. Output ONLY a single raw JSON object.\n'
        'WORKFLOW:\n'
        '1. {"tool": "list_candidates"}  ← only once if you need the list\n'
        '2. {"tool": "classify_candidate", "candidate_id": "c0", "confirmed": true}  ← for each\n'
        '3. {"tool": "advance_phase"}  ← after ALL classified\n\n'
        'CONFIRMED (confirmed:true) — these ARE personal PII:\n'
        '  ✅ Person names: "Grace Harris", "John Smith"\n'
        '  ✅ Email addresses: "john@gmail.com", "bob.davis@work.net"\n'
        '  ✅ Phone numbers: "555-123-4567", "438.565.1216", "+1-897-555-0123"\n'
        '  ✅ SSNs: "123-45-6789"\n\n'
        'REJECTED (confirmed:false) — these are NOT personal PII:\n'
        '  ❌ Dates: "2026-01-01", "2026-01-02"\n'
        '  ❌ IP addresses: "192.168.1.1", "10.0.0.5", "172.16.0.3"\n'
        '  ❌ System/test addresses: "test@example.com", "noreply@system.local", "admin@localhost"\n'
        '  ❌ Fake/test phones: "000.000.0000"\n'
        '  ❌ Log metadata: timestamps, INFO/ERROR/WARN, system paths\n\n'
        'RULE: Emails and phone numbers belonging to real people ARE PII — confirm them!'
    ),
    "REDACT": (
        'You are a PII compliance agent in REDACT phase. Output ONLY a single raw JSON object.\n'
        'WORKFLOW:\n'
        '1. {"tool": "redact_span", "candidate_id": "c0"}  ← for each CONFIRMED candidate\n'
        '2. {"tool": "submit"}  ← after ALL redacted\n\n'
        'Work through each confirmed candidate ID in order, then submit.'
    ),
}


# ---------------------------------------------------------------------------
# Episode state tracker
# ---------------------------------------------------------------------------
@dataclass
class EpisodeState:
    """Accumulates knowledge across steps so LLM gets rich context."""
    phase: str = "SCAN"
    files: list[str] = field(default_factory=list)
    files_read: dict[str, str] = field(default_factory=dict)
    candidates: dict[str, dict] = field(default_factory=dict)
    candidates_listed: bool = False
    redacted: list[str] = field(default_factory=list)
    last_tool_called: str = ""
    loop_counter: dict[str, int] = field(default_factory=dict)

    def absorb(self, action_json: str, result: str, phase: str) -> None:
        self.phase = phase
        try:
            parsed = _json.loads(action_json)
            tool = parsed.get("tool", "")
        except Exception:
            return

        self.last_tool_called = tool
        self.loop_counter[tool] = self.loop_counter.get(tool, 0) + 1

        if tool == "list_files" and "Files:" in result:
            try:
                raw = result.split("Files:")[1].strip()
                self.files = _json.loads(raw.replace("'", '"'))
            except Exception:
                pass

        elif tool == "read_file":
            fp = parsed.get("file_path", "")
            if fp and "===" in result:
                content = result.split("\n", 1)[1] if "\n" in result else result
                self.files_read[fp] = content[:1000]

        elif tool == "flag_candidate":
            m = re.search(r"Flagged (c\d+): (\w+) \| '(.+)'", result)
            if m:
                cid, pii_type, text = m.group(1), m.group(2), m.group(3)
                self.candidates[cid] = {"text": text, "pii_type": pii_type, "confirmed": None}

        elif tool == "list_candidates":
            self.candidates_listed = True
            for line in result.splitlines():
                m = re.search(r"(c\d+): \[(\w+)\] (\w+) \| '(.+)'", line)
                if m:
                    cid, status, pii_type, text = m.group(1), m.group(2), m.group(3), m.group(4)
                    confirmed = True if status == "CONFIRMED" else (False if status == "REJECTED" else None)
                    self.candidates[cid] = {"text": text, "pii_type": pii_type, "confirmed": confirmed}

        elif tool == "classify_candidate":
            cid = parsed.get("candidate_id", "")
            confirmed = parsed.get("confirmed")
            if cid in self.candidates:
                self.candidates[cid]["confirmed"] = bool(confirmed)

        elif tool == "redact_span":
            cid = parsed.get("candidate_id", "")
            if cid and cid not in self.redacted:
                self.redacted.append(cid)

    def build_context(self) -> str:
        lines = []

        if self.files:
            lines.append(f"FILES: {self.files}")
        if self.files_read:
            unread = [f for f in self.files if f not in self.files_read]
            lines.append(f"FILES READ: {list(self.files_read.keys())}")
            if unread:
                lines.append(f"FILES NOT YET READ: {unread}")
            for fname, content in self.files_read.items():
                lines.append(f"\n=== {fname} content ===\n{content[:800]}")

        if self.candidates:
            lines.append(f"\nFLAGGED CANDIDATES ({len(self.candidates)}):")
            for cid, info in self.candidates.items():
                status = {True: "CONFIRMED", False: "REJECTED", None: "PENDING"}.get(info.get("confirmed"))
                lines.append(f"  {cid}: [{status}] {info.get('pii_type','')} | {info.get('text','')!r}")

        if self.redacted:
            lines.append(f"REDACTED SO FAR: {self.redacted}")

        if self.loop_counter.get("list_files", 0) > 1:
            lines.append("\n!!! WARNING: You already called list_files. DO NOT call it again. READ THE FILES NEXT.")
        if self.loop_counter.get("list_candidates", 0) > 1:
            lines.append("\n!!! WARNING: You already called list_candidates. Classify each candidate now.")

        lines.append("\n=== WHAT TO DO NEXT ===")
        if self.phase == "SCAN":
            unread = [f for f in self.files if f not in self.files_read]
            if not self.files:
                lines.append("→ Call list_files to get the file list.")
            elif unread:
                lines.append(f"→ Call read_file for: {unread[0]}")
            else:
                # Detect PII in content
                found_pii = []
                for fname, content in self.files_read.items():
                    emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.]+', content)
                    phones = re.findall(r'[\+\d][\d\s\-\.\(\)]{8,}', content)
                    ssns = re.findall(r'\b\d{3}-\d{2}-\d{4}\b', content)
                    # Names: look for "user X Y logged in", "Contact: X Y", "User: X Y", etc.
                    names = re.findall(
                        r'(?:user|contact|User|Contact:|reach)\s+([A-Z][a-z]+ [A-Z][a-z]+)',
                        content
                    )
                    for pii in emails:
                        found_pii.append((pii.strip(), fname, "EMAIL"))
                    for pii in ssns:
                        found_pii.append((pii.strip(), fname, "SSN"))
                    for pii in names:
                        found_pii.append((pii.strip(), fname, "NAME"))
                    for pii in phones:
                        p = pii.strip()
                        if len(p) >= 10 and not any(p == e[0] for e in found_pii):
                            found_pii.append((p, fname, "PHONE"))
                flagged_texts = {v["text"] for v in self.candidates.values()}
                unflagged = [
                    (p, f, t) for p, f, t in found_pii
                    if not any(p in ft or ft in p for ft in flagged_texts)
                ]
                if unflagged:
                    p, f, ptype = unflagged[0]
                    lines.append(f'→ Call flag_candidate for: {{"tool": "flag_candidate", "text": "{p}", "file_path": "{f}", "pii_type": "{ptype}"}}')
                    if len(unflagged) > 1:
                        lines.append(f'  (and {len(unflagged)-1} more: {[x[0] for x in unflagged[1:3]]})')
                else:
                    lines.append(f"→ All PII flagged ({len(self.candidates)} candidates). Call advance_phase.")
        elif self.phase == "CLASSIFY":
            unclassified = [cid for cid, v in self.candidates.items() if v.get("confirmed") is None]
            if not self.candidates or (not self.candidates_listed and not unclassified):
                lines.append("→ Call list_candidates first.")
            elif unclassified:
                lines.append(f"→ Classify candidate {unclassified[0]}: classify_candidate with confirmed=true or false.")
            else:
                lines.append("→ All classified. Call advance_phase.")
        elif self.phase == "REDACT":
            confirmed = [cid for cid, v in self.candidates.items() if v.get("confirmed") is True]
            unredacted = [cid for cid in confirmed if cid not in self.redacted]
            if unredacted:
                lines.append(f"→ Redact {unredacted[0]}: redact_span with candidate_id={unredacted[0]!r}.")
            else:
                lines.append("→ All confirmed candidates redacted. Call submit.")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------
def extract_json(text: str) -> str:
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
                candidate = text[start : i + 1]
                try:
                    parsed = _json.loads(candidate)
                    if "action" in parsed and "tool" not in parsed:
                        parsed["tool"] = parsed.pop("action")
                    if "params" in parsed:
                        params = parsed.pop("params")
                        parsed.update(params)
                    if "filename" in parsed and "file_path" not in parsed:
                        parsed["file_path"] = parsed.pop("filename")
                    return _json.dumps(parsed)
                except _json.JSONDecodeError:
                    start = -1
                    depth = 0
    return '{"tool": "submit"}'


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def call_llm(client: OpenAI, phase: str, obs_result: str, state: EpisodeState, step: int) -> str:
    system = PHASE_SYSTEM[phase]
    context = state.build_context()

    user_msg = (
        f"Phase: {phase} | Step: {step}/{MAX_STEPS}\n"
        f"Last server response: {obs_result[:300]}\n\n"
        f"{context}\n\n"
        f"Allowed tools: {PHASE_TOOLS[phase]}\n"
        "Output ONE raw JSON object:"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        action = extract_json(raw)
        print(f"    [LLM] → {action}", flush=True)
        return action
    except Exception as e:
        print(f"[LLM ERROR] {e}", flush=True)
        return '{"tool": "submit"}'


# ---------------------------------------------------------------------------
# Single episode (runs env in-process, no HTTP)
# ---------------------------------------------------------------------------
def run_episode(
    client: OpenAI,
    level: int,
    episode_idx: int,
    seed: int | None = None,
) -> dict[str, Any]:
    env = ComplianceGuardEnv()
    obs = env.reset(level=level, seed=seed)
    phase = obs.agent_phase
    state = EpisodeState(phase=phase, files=list(obs.files_in_scope))
    final_reward = 0.0
    done = False
    steps = 0

    consecutive_loops = 0  # detect model stuck repeating same action

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        phase = obs.agent_phase
        state.phase = phase

        action_json = call_llm(client, phase, obs.last_action_result, state, step)

        # Loop guard: if env says "Already flagged", inject advance_phase after 2 retries
        if "Already flagged" in obs.last_action_result or "already" in obs.last_action_result.lower():
            consecutive_loops += 1
            if consecutive_loops >= 2:
                print(f"    [LOOP GUARD] Model stuck — injecting advance_phase", flush=True)
                action_json = '{"tool": "advance_phase"}'
                consecutive_loops = 0
        else:
            consecutive_loops = 0

        action = DataPrivacyAction(message=action_json)
        obs = env.step(action)
        reward = obs.last_reward
        done = obs.done
        steps = step
        if done:
            final_reward = obs.reward

        state.absorb(action_json, obs.last_action_result, obs.agent_phase)

        print(
            f"  ep={episode_idx} step={step} phase={phase} "
            f"reward={reward:+.4f} done={done} | {obs.last_action_result[:80]}",
            flush=True,
        )

    env.close()
    return {
        "episode": episode_idx,
        "level": level,
        "steps": steps,
        "reward": final_reward,
        "success": final_reward >= 0.7,
    }


# ---------------------------------------------------------------------------
# Baseline run
# ---------------------------------------------------------------------------
def run_baseline(levels: list[int], n_episodes: int, seeded: bool = False) -> None:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    results: list[dict] = []

    ep = 0
    for level in levels:
        for i in range(n_episodes):
            seed = i if seeded else None
            print(f"\n--- Episode {ep} | Level {level} | seed={seed} ---", flush=True)
            result = run_episode(client, level, ep, seed=seed)
            results.append(result)
            ep += 1
            time.sleep(0.3)  # rate limit guard

    successes = sum(1 for r in results if r["success"])
    success_rate = successes / len(results) if results else 0
    avg_reward = sum(r["reward"] for r in results) / len(results) if results else 0

    gate = "GREEN" if success_rate >= 0.3 else ("YELLOW" if success_rate >= 0.1 else "RED")

    summary = {
        "model": MODEL_NAME,
        "n_episodes": len(results),
        "success_rate": round(success_rate, 4),
        "avg_reward": round(avg_reward, 4),
        "gate": gate,
        "seeded": seeded,
        "results": results,
    }

    out_path = "baseline_results_seeded.json" if seeded else "baseline_results.json"
    with open(out_path, "w") as f:
        _json.dump(summary, f, indent=2)

    print(f"\n{'='*50}", flush=True)
    print(f"BASELINE COMPLETE", flush=True)
    print(f"  Model: {MODEL_NAME}", flush=True)
    print(f"  Episodes: {len(results)}", flush=True)
    print(f"  Success rate: {success_rate:.1%}", flush=True)
    print(f"  Avg reward: {avg_reward:.4f}", flush=True)
    print(f"  Gate: {gate}", flush=True)
    print(f"  Results saved: {out_path}", flush=True)
    if gate == "RED":
        print("  → RED: Consider SFT warmup before GRPO", flush=True)
    elif gate == "YELLOW":
        print("  → YELLOW: GRPO can proceed, expect slow start", flush=True)
    else:
        print("  → GREEN: Proceed directly to GRPO training", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=None, help="Single level (1-4)")
    parser.add_argument("--episodes", type=int, default=15, help="Episodes per level")
    parser.add_argument("--seeded", action="store_true", help="Use seed=i for episode i (produces baseline_results_seeded.json)")
    args = parser.parse_args()

    levels = [args.level] if args.level else [1, 3]
    run_baseline(levels=levels, n_episodes=args.episodes, seeded=args.seeded)
