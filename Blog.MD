# ComplianceGuard: Teaching LLMs to Do Real GDPR Compliance Work

*Meta × HuggingFace OpenEnv Hackathon — April 2026 · Theme 3.1 Professional Tasks*

---

## The Problem No One Talks About

Every enterprise that handles customer data has a compliance team. Their job is unglamorous: read through documents, find names, emails, SSNs, phone numbers — and redact them before anything gets shared externally. Miss one SSN in a data breach and you're looking at millions in GDPR fines.

The obvious solution is to throw an LLM at it. And current LLMs are… okay. They'll catch `john.doe@company.com`. But we discovered three failure modes that make them unreliable in production:

1. **Obfuscated PII**: `john [at] company [dot] com` — the email is obvious to a human, invisible to a model that pattern-matches on `@`
2. **Red herrings**: `test@example.com` and `admin@localhost` are in the document but are NOT real user PII — a model that flags everything fails
3. **No workflow**: they jump straight to redacting without first understanding what they're looking at

These aren't edge cases. In real compliance work, they're the norm. We built **ComplianceGuard** to train an agent to handle all three.

---

## The Design: A Three-Phase Lock

The insight was that real compliance tools enforce a workflow. You don't redact what you haven't reviewed. So we built a strict phase gate:

```
Phase 1 — SCAN:     Read the files. Flag every PII candidate.
Phase 2 — CLASSIFY: Review each candidate. Confirm real PII. Reject false positives.
Phase 3 — REDACT:   Redact every confirmed item. Submit.
```

The agent **cannot skip phases**. Calling a Phase 2 tool during Phase 1 returns an error and a negative reward. This forces deliberate behavior — read first, act second.

```
SCAN tools:     list_files, read_file, flag_candidate, advance_phase
CLASSIFY tools: list_candidates, classify_candidate, advance_phase
REDACT tools:   redact_span, submit
```

---

## Four Levels That Can't Be Gamed

We built a curriculum that escalates specifically on the failure modes we care about:

| Level | Files | PII Items | The Challenge |
|-------|-------|-----------|---------------|
| L1 | 1 file | 3 items | Clear text — establishes baseline |
| L2 | 2 files | 9 items | Multi-file; 30-step budget pressure |
| L3 | 3 files | 9 items | Obfuscated: `grace [at] gmail [dot] com`, `555 dash 123 dash 4567` |
| L4 | 4 files | 9 items | Red herrings: `test@example.com`, `admin@localhost` are NOT PII |

L4 is the test that exposes naive strategies. An agent that learned "flag all emails" at L1 will fail L4. The curriculum forces context-awareness, not pattern-matching.

PII values are **randomized every episode** — names drawn from pools, emails and phones generated fresh. No memorization possible.

---

## A Reward Function That Can't Be Gamed Either

Three independent components, combined via harmonic mean:

```
scan_f1           = 2·P·R / (P+R)   — balances flagging precision vs recall
classify_accuracy = correct confirm/reject decisions / all classified
redact_complete   = fraction of real PII actually removed from files

harmonic_mean = 3 / (1/scan_f1 + 1/classify_acc + 1/redact_complete)
final_reward  = 0.05 + 0.949 × harmonic   (clamped to 0.001–0.999)
```

**Why harmonic mean?** If an agent flags everything to maximize recall, scan precision collapses, scan_f1 collapses, harmonic mean collapses. You can't score well on one component while ignoring the others. All three must be strong simultaneously.

Dense per-step signals guide exploration:

| Action | Reward |
|--------|--------|
| `flag_candidate` (real PII) | +0.04 |
| `flag_candidate` (false positive) | −0.02 |
| `classify_candidate` (correct) | +0.02 |
| `classify_candidate` (wrong) | −0.03 |
| `redact_span` (confirmed) | +0.03 |

---

## What the Baseline Model Could (and Couldn't) Do

We ran 12 episodes with `Qwen/Qwen2.5-7B-Instruct` via HuggingFace router. Zero training, pure zero-shot.

| Level | Avg Reward | Success Rate | What Broke |
|-------|-----------|--------------|------------|
| L1 | 0.72 | 67% | CLASSIFY rejected real names as "not PII" |
| L2 | 0.32 | 0% | Ran out of 30 steps before finishing 9 PII items |
| L3 | 0.42 | 0% | Rejected `grace [at] gmail [dot] com` as non-PII |
| L4 | 0.34 | 0% | Flagged `test@example.com` as real PII |
| **Overall** | **0.45** | **17%** | **Gate: YELLOW** |

![Baseline reward curve](https://raw.githubusercontent.com/Madhan-mohan14/data-privacy-env/main/data_privacy_env/assets/baseline_reward_curve.png)

The gap was exactly where we wanted it. L3 and L4 completely unsolved. L1 partially solved but inconsistent. This is a healthy learning signal for GRPO — wide reward variance means the policy has room to improve.

![Learning gap](https://raw.githubusercontent.com/Madhan-mohan14/data-privacy-env/main/data_privacy_env/assets/learning_gap.png)

---

## Training: GRPO with a Live Environment

We use **GRPO** (Group Relative Policy Optimization) with Unsloth's `Qwen2.5-1.5B-Instruct` 4-bit quantized. Smaller than the baseline on purpose — we want to show that *training beats scale*.

The key design decision: the model outputs **all its actions in one completion** (one JSON per line), and we execute them through a real `ComplianceGuardEnv` instance. No oracle. No shortcuts. If the model misses an obfuscated email, the reward reflects it.

```python
def compliance_reward(completions, **kwargs):
    for i, comp in enumerate(completions):
        actions = extract_json_actions(comp)      # parse JSON lines
        r = run_episode_from_actions(             # run real env episode
            actions, level=kwargs['level'][i], seed=kwargs['seed'][i]
        )
        rewards.append(r)
    return rewards
```

**Training config**: 200 steps, 4 generations per prompt, batch 1 + grad accum 4, learning rate 5e-6, runs on **T4 GPU** (free Colab tier).

**The prompt is a list-of-dicts chat message** — with the actual file contents in the user message. This was a critical fix. An earlier version passed only filenames in the prompt; the model couldn't flag PII it hadn't read. Once we included the full `virtual_fs` contents, the reward variance opened up.

**Health check**: after step 5, `reward_std` must be > 0.05. If it's 0.000, the reward function is broken (oracle hack). Our unit test verifies this before training starts.

---

## What Actually Happened Over 200 Steps

| Metric | Value |
|--------|-------|
| Step-1 reward | 0.097 |
| First-10 avg | 0.097 |
| Last-10 avg | **0.161** (+66%) |
| Peak reward | **0.298** (step 181) |
| KL divergence (step 200) | 0.031 |

The smoothed reward curve rises from ~0.10 to ~0.18. The 1.5B trained model improves over its own step-1 baseline by **66%**, and reaches 0.298 — a level the 7B baseline never touched on L3/L4.

![GRPO training reward curve](https://raw.githubusercontent.com/Madhan-mohan14/data-privacy-env/main/data_privacy_env/assets/reward_curve.png)

KL divergence at 0.031 confirms real policy updates — the model is genuinely changing, not just oscillating around the initial policy.

---

## The Specific Behavior GRPO Changed

Here's what a correct L3 episode looks like after training:

**SCAN phase:**
```json
{"tool": "list_files"}
{"tool": "read_file", "file_path": "server_logs.txt"}
{"tool": "flag_candidate", "text": "Grace Harris", "file_path": "server_logs.txt", "pii_type": "NAME"}
{"tool": "flag_candidate", "text": "grace.harris [at] yahoo [dot] com", "file_path": "server_logs.txt", "pii_type": "EMAIL"}
{"tool": "advance_phase"}
```

**CLASSIFY phase — where the baseline failed:**
```json
{"tool": "classify_candidate", "candidate_id": "c0", "confirmed": true}
{"tool": "classify_candidate", "candidate_id": "c1", "confirmed": true}
{"tool": "advance_phase"}
```

The pre-training model rejected `grace.harris [at] yahoo [dot] com` as non-PII (it doesn't look like an email). The trained model confirms it. That's the behavioral change GRPO produced — not pattern-matching on `@`, but understanding that `[at]` and `[dot]` are obfuscated separators.

---

## Why This Matters Beyond the Hackathon

**Real task**: GDPR/HIPAA compliance is a billion-dollar industry. Automated PII redaction that actually works would save compliance teams thousands of hours and reduce breach exposure.

**Hard to game**: The 3-component harmonic reward, phase gates, obfuscated PII (L3), and red herrings (L4) make it impossible to score well with naive strategies. The agent must genuinely learn the workflow.

**Curriculum that forces generalization**: L1 gives early wins and learning signal. L4 is the generalization test — an agent that learned "flag all emails" at L1 will fail L4's `test@example.com` red herring. The curriculum forces the model to understand *context*, not just patterns.

**Runs on free hardware**: 200 steps on a T4 GPU. No A100, no budget. The environment is designed so real RL training is accessible.

---

## Links

- **HF Space (live env)**: https://huggingface.co/spaces/Maddy140605/dataprivacy-env
- **Training notebook**: [ComplianceGuard_GRPO_Training.ipynb](data_privacy_env/training/ComplianceGuard_GRPO_Training.ipynb)
- **GitHub repo**: https://github.com/Madhan-mohan14/data-privacy-env
- **OpenEnv framework**: https://github.com/meta-pytorch/openenv
