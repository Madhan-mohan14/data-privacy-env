# ComplianceGuard: Teaching LLMs to Do Real GDPR Compliance Work

*Meta × HuggingFace OpenEnv Hackathon — April 2026*

---

## The Problem

Every enterprise that handles customer data has a GDPR/HIPAA compliance team.
Their job: find and redact PII (names, emails, SSNs, phone numbers) from
documents before they leave the organization. It's tedious, error-prone, and
expensive.

Current LLMs are inconsistently good at this. They can spot an obvious
`john.doe@company.com` but they miss obfuscated forms like
`john[at]company.com`, struggle with multi-file tasks, and hallucinate
false positives on things like IP addresses.

We built **ComplianceGuard** — an RL environment that trains agents to do
this properly.

---

## The Environment

ComplianceGuard enforces the same 3-phase workflow a real compliance tool uses:

**Phase 1 — SCAN**: Read files, flag every PII candidate you find.  
**Phase 2 — CLASSIFY**: Review candidates, confirm real PII, reject false
positives.  
**Phase 3 — REDACT**: Redact every confirmed candidate, then submit.

The agent cannot skip phases. Calling a Phase 2 tool during Phase 1 returns
an error. This *forces* the agent to build a mental model of the document
before acting — instead of pattern-matching its way to a guess.

### The Curriculum

We built 4 levels of increasing difficulty:

| Level | What makes it hard |
|-------|-------------------|
| L1 | One file, 3 clear PII items |
| L2 | CSV database — PII in specific columns only |
| L3 | Obfuscated PII: `john[at]company.com`, `555 dash 123 dash 4567` |
| L4 | Red herrings: `test@example.com` is in the file but is NOT PII |

L4 is the key challenge. An agent that just flags every email address will
fail — it needs to understand context to distinguish real user PII from
system addresses.

### The Reward Signal

```
if perfect (recall ≥ 0.99, precision ≥ 0.99, completeness ≥ 0.99):
    reward = 0.999

elif partial (recall ≥ 0.50, completeness > 0):
    reward = 0.3 + 0.6 × (recall × precision × completeness)

else:
    reward = 0.05
```

The product term is key: high recall with low precision gets you 0.3 + almost
nothing. You need all three to get a good score. This prevents the agent from
taking the easy path of flagging everything.

---

## Baseline: What the Model Can Do Without Training

We ran 30 episodes with `llama-3.3-70b-versatile` (Groq) before any training:

| | L1 (easy) | L3 (hard) |
|---|---|---|
| Success rate | **86.7%** | **0%** |
| Avg reward | 0.92 | 0.05 |

L1 is nearly solved. L3 is completely unsolved. The model can find obvious
PII but fails entirely on obfuscated forms.

**This gap is exactly what we want to train on.**

---

## Training: GRPO with Live Environment

We used **GRPO** (Group Relative Policy Optimization) with Unsloth's
`Qwen3-1.7B` 4-bit quantized model. The key is `environment_factory` — the
training loop talks to a live `ComplianceGuardEnv` instance, not a static
dataset. The model generates actions, the environment responds, the reward
shapes the next gradient.

```python
trainer = GRPOTrainer(
    model=model,
    config=GRPOConfig(max_steps=200, num_generations=4, ...),
    train_dataset=curriculum_dataset,   # L1→L4 with seeds
    environment_factory=ComplianceGuardEnvTRL,
)
trainer.train()
```

### Results After Training

*(Training run completed onsite on HF A100 — results to be filled in)*

| | Before GRPO | After 200 steps GRPO |
|---|---|---|
| L1 reward | 0.92 | — |
| L3 reward | 0.05 | — |
| L4 reward | 0.05 | — |

---

## Why This Matters

- **Real task**: GDPR compliance is a billion-dollar industry. Training models
  to do it reliably has direct enterprise value.
- **Hard to game**: The product reward, phase gates, and L4 red herrings make
  it impossible to score well with naive strategies.
- **Curriculum design**: L1 gives the model early wins and learning signal.
  L4 is the generalization test. The curriculum forces the model to actually
  learn the workflow, not just memorize patterns.

---

## Links

- **Environment (HF Space)**: https://maddy140605-dataprivacy-env.hf.space
- **Training notebook**: `training/ComplianceGuard_GRPO_Training.ipynb`
- **Code repository**: *(GitHub link)*
- **Baseline results**: `baseline_results.json`
