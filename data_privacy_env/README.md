---
title: ComplianceGuard
emoji: 🔒
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - privacy
  - pii
  - gdpr
---

# ComplianceGuard — PII Redaction RL Environment

**Meta × HuggingFace OpenEnv Hackathon | Theme 3.1 — Professional Tasks**

An RL environment where an agent learns to perform real enterprise GDPR/HIPAA
compliance work: scan documents, classify sensitive data, and redact PII — in
the same three-phase workflow a human compliance officer follows.

---

## The Problem

Large language models are increasingly used in enterprise settings where they
handle documents containing PII (names, emails, SSNs, phone numbers). Getting
models to reliably find *and* redact *all* PII — without false positives —
requires multi-step reasoning that current models struggle with.

**ComplianceGuard** trains agents to solve this with a structured 3-phase
workflow, adaptive curriculum (L1→L4), and a rich reward signal that rewards
precision, recall, and completeness together.

---

## Environment Design

### Three-Phase Workflow: SCAN → CLASSIFY → REDACT

The agent cannot jump straight to redacting. It must work through phases in
order — exactly like real compliance tooling.

```
Phase 1 SCAN       Phase 2 CLASSIFY      Phase 3 REDACT
──────────────     ─────────────────     ──────────────
list_files    →    list_candidates  →    redact_span
read_file          classify_candidate    submit
flag_candidate     advance_phase
advance_phase
```

**Phase gate**: using a tool from the wrong phase returns an error and a
−0.05 penalty. This forces the agent to learn workflow discipline, not just
pattern matching.

### Curriculum: 4 Levels

| Level | Files | PII Items | Difficulty |
|-------|-------|-----------|------------|
| L1 | 1 server log | 3 (name, email, phone) | Baseline |
| L2 | 1 CSV database | 6 (structured data) | Medium |
| L3 | 2 files | 9 (obfuscated: `john[at]email.com`) | Hard |
| L4 | 3 files | 9 + red herrings (`test@example.com`) | Expert |

L4 introduces red herrings — fake email/phone addresses in the files that
are NOT real PII. The agent must classify correctly and avoid false positives.

### Reward Function (near-binary)

```
if scan_recall ≥ 0.99 AND precision ≥ 0.99 AND redact_completeness ≥ 0.99:
    reward = 0.999          ← perfect episode

elif scan_recall ≥ 0.50 AND redact_completeness > 0:
    reward = 0.3 + 0.6 × (scan_recall × precision × redact_completeness)

else:
    reward = 0.05           ← floor (missed everything)

reward = clamp(reward, 0.001, 0.999)   ← OpenEnv bounds
```

- **scan_recall**: fraction of real PII that was flagged as a candidate
- **precision**: fraction of confirmed candidates that are actually PII
- **redact_completeness**: fraction of real PII physically removed from files

All three must be high simultaneously — a high-recall but low-precision agent
gets punished by the product term.

---

## Baseline Results (llama-3.3-70b-versatile, no training)

30 episodes: 15 × L1, 15 × L3

| Metric | Value |
|--------|-------|
| L1 success rate (reward ≥ 0.7) | **86.7%** |
| L3 success rate | **0%** |
| Overall success rate | **43.3%** |
| Average reward | **0.43** |
| Baseline gate | **GREEN** |

L1 is solvable by the baseline model (clear PII, single file).  
L3 fails completely (obfuscated PII like `john[at]email.com` requires
pattern recognition the base model lacks).

**This gap is the training signal**: GRPO training on L1→L4 should bring
L3/L4 performance from 0% toward L1 levels.

---

## Training Pipeline

**Model**: `unsloth/Qwen3-1.7B` (4-bit quantized, LoRA r=16)  
**Algorithm**: GRPO via TRL `environment_factory` API  
**Compute**: HuggingFace A100 (onsite Apr 25-26)  
**Steps**: 200 GRPO steps, curriculum L1→L4

```python
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=grpo_config,
    train_dataset=dataset,
    environment_factory=ComplianceGuardEnvTRL,  # live env, not static dataset
)
```

The training notebook is at:
`training/ComplianceGuard_GRPO_Training.ipynb`

---

## Quick Start

```bash
# Install
uv sync

# Run server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run oracle baseline (5 episodes, no LLM needed)
python -c "
import json, sys
sys.path.insert(0, '.')
from server.data_privacy_env_environment import ComplianceGuardEnv
from models import DataPrivacyAction

def act(tool, **kw):
    return DataPrivacyAction(message=json.dumps({'tool': tool, **kw}))

env = ComplianceGuardEnv()
env.reset(level=1, seed=42)
for f in env.virtual_fs: env.step(act('read_file', file_path=f))
for pii in env.pii_list:
    for f,c in env.virtual_fs.items():
        if pii in c: env.step(act('flag_candidate', text=pii, file_path=f, pii_type='OTHER')); break
env.step(act('advance_phase'))
for cid in env.candidates: env.step(act('classify_candidate', candidate_id=cid, confirmed=True))
env.step(act('advance_phase'))
for cid,v in env.candidates.items():
    if v['confirmed']: env.step(act('redact_span', candidate_id=cid))
obs = env.step(act('submit'))
print(f'reward={obs.reward}')  # → 0.999
"

# Run Groq inference baseline
export GROQ_API_KEY=your_key
python inference.py --level 1 --episodes 5
```

---

## API Reference

**Action**: JSON string with `"tool"` key plus phase-specific fields.

| Tool | Phase | Required fields |
|------|-------|----------------|
| `list_files` | SCAN | — |
| `read_file` | SCAN | `file_path` |
| `flag_candidate` | SCAN | `text`, `pii_type` |
| `advance_phase` | SCAN / CLASSIFY | — |
| `list_candidates` | CLASSIFY | — |
| `classify_candidate` | CLASSIFY | `candidate_id`, `confirmed` (bool) |
| `redact_span` | REDACT | `candidate_id` |
| `submit` | REDACT | — |

**Observation** fields: `task_id`, `task_description`, `available_tools`,
`last_action_result`, `last_reward`, `cumulative_reward`, `files_in_scope`,
`step_number`, `max_steps`, `done`, `reward`, `agent_phase`,
`curriculum_level`, `candidate_count`, `classified_count`,
`last_candidate_id`, `metrics`.

---

## Repository Structure

```
data_privacy_env/
├── server/
│   ├── app.py                          # FastAPI server (factory pattern)
│   └── data_privacy_env_environment.py # ComplianceGuardEnv core
├── curriculum/
│   ├── generators.py                   # L1–L4 task generators
│   └── manager.py                      # Adaptive curriculum manager
├── training/
│   ├── grpo_env.py                     # TRL wrapper (reset()→str)
│   ├── grpo_train.py                   # Unsloth GRPO training script
│   ├── eval_checkpoint.py              # Checkpoint evaluation
│   ├── plan_b_groq.py                  # Groq-based fallback training
│   └── ComplianceGuard_GRPO_Training.ipynb  # Colab notebook
├── agents/prompts.py                   # Phase-specific LLM prompts
├── models.py                           # DataPrivacyAction / Observation
├── inference.py                        # Groq inference loop (in-process)
├── tests/test_env_local.py             # 12 unit tests (all pass)
├── baseline_results.json               # 30-episode Groq baseline
├── openenv.yaml                        # OpenEnv manifest
└── Dockerfile                          # HF Space deployment
```

---

## Docker

```bash
docker build -t complianceguard .
docker run -p 7860:7860 complianceguard
curl http://localhost:7860/health  # → {"status":"healthy"}
```

---

## Links

- **HF Space (live env)**: https://maddy140605-dataprivacy-env.hf.space
- **Training notebook**: `training/ComplianceGuard_GRPO_Training.ipynb`
- **Baseline results**: `baseline_results.json`
- **HF Blog post**: *(link after onsite training run)*
