# ComplianceGuard: Teaching LLMs to Do Real GDPR Compliance Work

*Meta × HuggingFace OpenEnv Hackathon — April 2026 · Theme 3.1 Professional Tasks*

---

## The Problem

Every enterprise that handles customer data has a compliance team. Their job: find and redact PII (names, emails, SSNs, phone numbers) from documents before they're shared externally. It's tedious, error-prone at scale, and a single missed SSN in a data breach can result in millions in GDPR fines.

Current LLMs are inconsistently good at this. They can spot an obvious `john.doe@company.com` but they:
- Miss obfuscated forms like `john [at] company [dot] com`
- Struggle with multi-file tasks where PII is spread across 3 files
- Hallucinate false positives — flagging `test@example.com` or `admin@localhost` as real PII
- Have no notion of *workflow* — they jump to redacting before fully reading the documents

We built **ComplianceGuard** — an RL environment that trains agents to do this properly, using the same 3-phase workflow a real compliance tool enforces.

---

## The Environment

### Three Phases, No Shortcuts

ComplianceGuard enforces a strict phase gate:

**Phase 1 — SCAN**: Read files, flag every PII candidate you see.
**Phase 2 — CLASSIFY**: Review each candidate — confirm real PII, reject false positives.
**Phase 3 — REDACT**: Redact every confirmed candidate, then submit.

The agent **cannot skip phases**. Calling a Phase 2 tool during Phase 1 returns an error and a penalty reward. This forces the agent to build a mental model of the document before acting, instead of guessing.

```
SCAN tools:    list_files, read_file, flag_candidate, advance_phase
CLASSIFY tools: list_candidates, classify_candidate, advance_phase
REDACT tools:   redact_span, submit
```

### Four Curriculum Levels

| Level | Files | PII Items | Challenge |
|-------|-------|-----------|-----------|
| L1 | 1 file | 3 items | Baseline — clear text, obvious patterns |
| L2 | 2 files | 9 items | Must read all files; step budget pressure |
| L3 | 3 files | 9 items | Obfuscated: `john [at] email [dot] com`, `555 dash 123 dash 4567` |
| L4 | 4 files | 9 items | Red herrings: `test@example.com`, `admin@localhost` in scope but NOT real PII |

L4 is the key challenge. An agent that just flags every email address will fail — it needs to understand context to distinguish real user PII from system/test addresses.

PII values are **randomized every episode** (names drawn from pools, emails and phones generated). The agent cannot memorize a fixed answer.

### The Reward Signal

Three independent components, combined via harmonic mean:

```
scan_f1           = 2·P·R / (P+R)   (flagging precision vs recall)
classify_accuracy = correct confirm/reject decisions / all classified
redact_complete   = fraction of real PII actually removed from files

harmonic_mean = 3 / (1/scan_f1 + 1/classify_acc + 1/redact_complete)
final_reward  = 0.05 + 0.949 × harmonic   (clamped to 0.001–0.999)
```

**Why harmonic mean?** It prevents gaming. An agent that flags everything achieves high recall but low precision — the harmonic mean of scan_f1 collapses, keeping the final reward low. All three components must be strong simultaneously.

Dense per-step rewards also provide immediate signal:
- `flag_candidate` real PII → +0.04
- `flag_candidate` false positive → −0.02
- `classify_candidate` correct → +0.02 / wrong → −0.03
- `redact_span` confirmed → +0.03

---

## Baseline: What the Model Can Do Without Training

We ran 12 episodes with `Qwen/Qwen2.5-7B-Instruct` via HuggingFace router. No fine-tuning, no training — pure baseline.

| Level | Avg Reward | Success Rate | Key Failure Mode |
|-------|-----------|--------------|-----------------|
| L1 | 0.72 | 67% | CLASSIFY rejects person names as not-PII |
| L2 | 0.32 | 0% | Runs out of 30 steps with 9 PII items |
| L3 | 0.42 | 0% | Cannot parse obfuscated `john [at] domain [dot] com` |
| L4 | 0.34 | 0% | Flags `test@example.com` as real PII (false positive) |
| **Overall** | **0.45** | **17%** | **Gate: YELLOW** |

![Baseline reward curve](https://raw.githubusercontent.com/Madhan-mohan14/data-privacy-env/main/data_privacy_env/assets/baseline_reward_curve.png)

The gap is exactly where we want to train. L3 and L4 are completely unsolved — the model has no concept of obfuscated PII or red herrings. L1 is partially solved but inconsistent. This gives GRPO a rich signal to work with.

![Learning gap](https://raw.githubusercontent.com/Madhan-mohan14/data-privacy-env/main/data_privacy_env/assets/learning_gap.png)

---

## Training: GRPO with Live Environment

We use **GRPO** (Group Relative Policy Optimization) with Unsloth's `Qwen2.5-1.5B-Instruct` 4-bit quantized. The model is smaller than the baseline agent on purpose — we want to show that *training* beats *scale*.

**The key design**: the model outputs all its actions in one completion (one JSON per line), and we execute them through a real `ComplianceGuardEnv` instance. No oracle, no shortcuts — if the model misses a PII item, the reward reflects it.

```python
def compliance_reward(completions, **kwargs):
    for i, comp in enumerate(completions):
        actions = extract_json_actions(comp)     # parse JSON lines
        r = run_episode_from_actions(            # run real env episode
            actions, level=kwargs['level'][i], seed=kwargs['seed'][i]
        )
        rewards.append(r)
    return rewards
```

**Training config**: 200 steps, 4 generations per prompt, batch 2 + grad accum 4, learning rate 5e-6, runs on **T4 GPU** (free Colab tier).

**Health check**: after step 5, `reward_std` must be > 0.05. If it's 0.000, the reward function is broken (oracle hack). Our unit test verifies this before training starts.

### Results After Training

*(Training notebook: [`training/ComplianceGuard_GRPO_Training.ipynb`](data_privacy_env/training/ComplianceGuard_GRPO_Training.ipynb))*

| | Before GRPO | After 200 steps GRPO | Improvement |
|---|---|---|---|
| L1 reward | 0.72 | — | — |
| L2 reward | 0.32 | — | — |
| L3 reward | 0.42 | — | — |
| L4 reward | 0.34 | — | — |

*Training curves (reward_curve.png) will be added here after the Colab run completes.*

---

## Why This Matters

**Real task**: GDPR/HIPAA compliance is a billion-dollar industry. Automated PII redaction that actually works would save compliance teams thousands of hours and reduce breach exposure.

**Hard to game**: The 3-component harmonic reward, phase gates, obfuscated PII (L3), and red herrings (L4) make it impossible to score well with naive strategies. The agent must genuinely learn the workflow.

**Curriculum that forces generalization**: L1 gives early wins and learning signal. L4 is the generalization test — an agent that learned to "flag all emails" at L1 will fail L4's `test@example.com` red herring. The curriculum forces the model to understand *context*, not just patterns.

**Real tool interaction**: Unlike most RL environments, the agent interacts with a multi-file virtual filesystem through typed tools. It must chain observations across steps — read a file, remember what it found, flag the right spans, classify correctly, then redact only confirmed items.

---

## Links

- **HF Space**: https://huggingface.co/spaces/Maddy140605/dataprivacy-env
- **Training notebook**: [ComplianceGuard_GRPO_Training.ipynb](data_privacy_env/training/ComplianceGuard_GRPO_Training.ipynb)
- **GitHub repo**: https://github.com/Madhan-mohan14/data-privacy-env
- **OpenEnv framework**: https://github.com/meta-pytorch/openenv
