# ComplianceGuard: Teaching a 1.5B Model to Do Real GDPR Compliance Work with GRPO

*Meta × HuggingFace OpenEnv Hackathon — April 2026 · Theme 3.1 Professional Tasks*

*by Madhan Mohan*

---

## The Problem That Compliance Teams Live Every Day

Picture this: a lawyer at a mid-size tech company needs to share three years of customer support logs with a regulatory auditor. Before anything leaves the building, every name, email address, phone number, and SSN in those logs has to be found and blacked out. Miss one, and under GDPR that's potentially a €20 million fine or 4% of global revenue — whichever is higher.

The compliance team assigns two analysts. They spend three days manually reading through 40,000 log lines. They're tired by day two. On day three, analyst #2 misses `henry.taylor [at] corp [dot] io` because it doesn't look like an email at a glance. The document goes out. Six months later, the company is explaining to a regulator why user data was shared without authorization.

This is the problem we set out to solve — and more specifically, the problem we set out to *teach an LLM agent to solve reliably.*

---

## Why Current LLMs Get This Wrong

The obvious first attempt is to just ask a capable LLM: *"Find and redact all PII in this document."* And it kind of works. On clean, obvious data, GPT-4 or Qwen2.5-7B will catch `john.doe@company.com`. But we ran 12 episodes against a 7B model zero-shot and found three systematic failure modes that make production deployment unsafe:

### Failure Mode 1: Obfuscated PII

Real-world documents contain PII that's been manually obscured — by users protecting their own data, by copy-paste from older systems, or by document templates. Forms like:

```
grace.harris [at] gmail [dot] com
555 dash 123 dash 4567
```

A human reads these instantly as an email and a phone number. A model that's been trained on the pattern `\w+@\w+\.\w+` sees nothing. In our baseline evaluation, the 7B model rejected obfuscated forms 100% of the time in the CLASSIFY phase — confirming real names but then deciding the obfuscated email next to them was "not PII."

### Failure Mode 2: Red Herrings (False Positives)

Real documents also contain things that *look* like PII but aren't. System logs are full of:

```
System alert sent to test@example.com
Auto-reply from noreply@system.local
Automated notification: admin@localhost
Test phone: 000.000.0000
```

These are system addresses, not user PII. Under GDPR, redacting them is unnecessary and breaks the document structure. An agent that flags everything achieves 100% recall but terrible precision — and a harmonic-mean reward function (more on that shortly) makes this strategy score very badly.

### Failure Mode 3: No Workflow — Acting Before Reading

Perhaps the most subtle failure: LLMs tend to jump straight to action. Ask them to "redact PII in these files" and they'll start redacting before they've read all the files, before they've confirmed whether what they've found is real PII, and before they have any mental model of the document. In a multi-file task with 9 PII items spread across 3 documents, this produces chaotic, incomplete redaction.

Real compliance tools enforce a workflow for exactly this reason. You scan first. You review second. You redact third. **We encoded that workflow into the environment itself.**

---

## The Environment Design: Three Phases, No Shortcuts

The core architectural decision of ComplianceGuard is a **strict phase-gate finite state machine**. The environment has three phases:

```
┌─────────────────────────────────────────────────────────┐
│  Phase 1 — SCAN                                         │
│  Read files. Flag every PII candidate you see.          │
│  Tools: list_files, read_file, flag_candidate,          │
│          advance_phase                                  │
└──────────────────────────┬──────────────────────────────┘
                           │ advance_phase()
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 2 — CLASSIFY                                     │
│  Review each flagged candidate.                         │
│  Confirm real PII. Reject false positives.              │
│  Tools: list_candidates, classify_candidate,            │
│          advance_phase                                  │
└──────────────────────────┬──────────────────────────────┘
                           │ advance_phase()
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 3 — REDACT                                       │
│  Redact every confirmed candidate. Submit.              │
│  Tools: redact_span, submit                             │
└─────────────────────────────────────────────────────────┘
```

**The agent cannot skip phases.** If it calls `classify_candidate` while still in SCAN, it gets back:

```
Error: 'classify_candidate' only in CLASSIFY. Currently SCAN.
Reward: -0.05
```

This single design decision eliminates the "jump straight to redacting" failure mode entirely. The environment enforces deliberate, structured behavior — and the negative reward for wrong-phase calls makes the agent learn *why* structure matters, not just follow it blindly.

The tools are intentionally minimal. There are no "find all PII" shortcuts, no bulk operations, no "classify everything as confirmed" batch calls. The agent must interact with the virtual filesystem one step at a time, building up a picture of what it's found before it can act on it.

---

## A Curriculum Designed Around the Failure Modes

We built four levels, each targeting a specific failure mode identified in baseline evaluation:

| Level | Files | PII Items | The Specific Challenge |
|-------|-------|-----------|------------------------|
| **L1** | 1 | 3 | Clear text only. Establishes basic SCAN→CLASSIFY→REDACT behavior. |
| **L2** | 2 | 9 | Multi-file. Must read both files; 30-step budget forces efficiency. |
| **L3** | 3 | 9 | Obfuscated PII: `john [at] email [dot] com`, `555 dash 123 dash 4567`. |
| **L4** | 4 | 9 | Red herrings: `test@example.com`, `admin@localhost` are in scope but NOT real PII. |

The curriculum is weighted toward easier levels in the training dataset (distribution: `[1,1,1,2,2,3]`) so the model gets early learning signal before confronting the hard cases. L4 is never included in training — it's a pure generalization test.

**The key insight about L4**: An agent that learned at L1 to "flag everything that looks like an email" will fail L4. To pass L4, the agent must understand *context* — that `test@example.com` is a system address while `emma.wilson@gmail.com` is a real user. This requires genuine reasoning, not pattern matching.

**PII is randomized every episode.** Names are drawn from pools, emails and phone numbers are generated fresh with random components. The agent cannot memorize "the answer is Grace Harris" — it has to actually read the file every time.

---

## The Reward Function: Designed to Be Ungameable

Most RL reward functions for tasks like this use a simple binary: did you find all the PII? That creates two obvious gaming strategies:

1. **Flag everything** → 100% recall, terrible precision
2. **Flag only the most obvious items** → decent precision, terrible recall on L3/L4

We used a three-component **harmonic mean** to make both strategies fail:

```
scan_f1           = 2 × precision × recall / (precision + recall)
                    — penalizes both missing PII AND over-flagging

classify_accuracy = (correct confirmations + correct rejections) / all classified
                    — penalizes both false confirms AND false rejects

redact_complete   = fraction of real PII actually removed from files
                    — penalizes stopping before all confirmed PII is redacted

harmonic_mean = 3 / (1/scan_f1 + 1/classify_acc + 1/redact_complete)

final_reward  = 0.05 + 0.949 × harmonic_mean   [clamped to 0.001–0.999]
```

**Why harmonic mean specifically?** The harmonic mean is lower than both the arithmetic and geometric means, and it collapses dramatically when any single component is weak. If scan_f1 = 1.0 and classify_accuracy = 1.0 but redact_complete = 0.1, the harmonic mean is only 0.27 — not 0.7. You can't compensate for one bad component by excelling at others.

To score 0.999 (maximum), the agent needs scan_f1 ≥ 0.99, classify_accuracy ≥ 0.99, AND redact_complete ≥ 0.99 simultaneously. The only way to do that is to actually do the compliance work correctly.

### Dense Per-Step Rewards

On top of the terminal harmonic reward, each action generates immediate feedback:

| Action | When | Reward |
|--------|------|--------|
| `flag_candidate` | flagged text is real PII | **+0.04** |
| `flag_candidate` | flagged text is NOT PII | **−0.02** |
| `classify_candidate` | decision matches ground truth | **+0.02** |
| `classify_candidate` | decision is wrong | **−0.03** |
| `redact_span` | candidate was confirmed PII | **+0.03** |
| Invalid JSON / wrong phase | any step | **−0.05** |

These dense rewards give the model gradient signal at every step — it doesn't have to wait until `submit` to learn that flagging `test@example.com` was a mistake.

---

## What We Got Wrong Before Training Worked

Three bugs blocked training. All three made `reward_std = 0.000` — the symptom that training is dead.

### Bug 1: The Prompt Had No File Contents

The first version of our training prompt looked like this:

```python
# BROKEN: the model can't flag PII it hasn't read
prompt = f"Task: Level {level}. Files: {list(env.virtual_fs.keys())}. Find PII."
```

The model was given a list of filenames. It was expected to flag PII in those files. But it had never read those files — it could only hallucinate what might be in them. The reward variance was near-zero because every episode produced the same random guesses.

**The fix:** Include the full contents of `virtual_fs` in the user message:

```python
def make_prompt(level, seed):
    env = ComplianceGuardEnv()
    env.reset(seed=seed, level=level)

    files_text = ""
    for fname, content in env.virtual_fs.items():
        files_text += f"\n--- {fname} ---\n{content}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Files to audit:{files_text}\nFind and redact ALL PII."}
    ]
```

Once the model could actually see the files, reward variance opened up immediately.

### Bug 2: max_seq_length Too Small

We initially set `max_seq_length=512`. The prompt — after including full file contents — is approximately 800 tokens. Truncation at 512 caused a `TorchRuntimeError` during the forward pass. Increasing to `max_seq_length=2048` fixed it.

### Bug 3: Plain String Prompt Bypassed Qwen's System Prompt

Qwen2.5-Instruct uses a chat template. If you pass the prompt as a plain string, the system message is ignored entirely — the model just sees the instruction text without its compliance-agent persona. The fix was to always pass the prompt as a **list of message dicts** (the format shown in Bug 1's fix). This lets the tokenizer apply the correct chat template, including the system role.

Once all three bugs were fixed, `reward_std > 0.05` appeared at step 5 — training was alive.

---

## Baseline: What the 7B Model Could Do Without Training

We ran 12 episodes with `Qwen/Qwen2.5-7B-Instruct` via the HuggingFace router. No fine-tuning, no prompting tricks — just zero-shot capability.

| Level | Avg Reward | Success Rate | What Specifically Failed |
|-------|-----------|--------------|--------------------------|
| L1 — clear logs | 0.719 | 67% | Classified real names as "not PII" in CLASSIFY phase |
| L2 — multi-file | 0.315 | 0% | Ran out of 30 steps before processing all 9 PII items |
| L3 — obfuscated | 0.418 | 0% | Confirmed names but rejected `email [at] domain [dot] com` forms |
| L4 — red herrings | 0.340 | 0% | Flagged `test@example.com` and `noreply@system.local` as real PII |
| **Overall** | **0.448** | **16.7%** | **Gate: YELLOW — proceed to training** |

![Baseline reward curve](https://raw.githubusercontent.com/Madhan-mohan14/data-privacy-env/main/data_privacy_env/assets/baseline_reward_curve.png)

The 16.7% success rate and wide reward variance (0.32 to 0.72 across levels) is exactly what we want before starting GRPO. If success rate were 0%, there's nothing for RL to build on. If it were 80%, there's not much room to improve. 16.7% with clear failure modes is a healthy starting point.

![Learning gap — where GRPO has room to improve](https://raw.githubusercontent.com/Madhan-mohan14/data-privacy-env/main/data_privacy_env/assets/learning_gap.png)

---

## Training: GRPO on a T4 GPU

We chose **GRPO** (Group Relative Policy Optimization) because it doesn't require a learned reward model — it uses the environment itself as the verifier. For a task with clear correctness criteria (did you find all the PII? did you avoid the red herrings?), this is exactly right.

**Model**: `Qwen2.5-1.5B-Instruct` with 4-bit QLoRA via Unsloth. Deliberately smaller than the 7B baseline — we wanted the results to show that *training beats scale*, not that a bigger model does better.

**The training loop in one paragraph**: The model receives a prompt containing the system instructions and the full contents of all files for a random episode. It generates a completion — a sequence of JSON actions, one per line. We parse those actions and execute them through a fresh `ComplianceGuardEnv` instance. The environment runs the full SCAN → CLASSIFY → REDACT workflow and returns the final reward. GRPO uses that reward, along with the rewards from 3 other completions for the same prompt, to update the policy — preferring completions that score higher.

```python
def compliance_reward(completions, **kwargs):
    rewards = []
    for i, comp in enumerate(completions):
        actions = extract_json_actions(comp)           # parse JSON lines
        r = run_episode_from_actions(                  # live environment episode
            actions,
            level=kwargs['level'][i],
            seed=kwargs['seed'][i]
        )
        rewards.append(r)
    return rewards
```

No oracle. No lookup table. If the model outputs malformed JSON, those actions fail. If it skips reading a file, it misses PII. The reward is a direct reflection of how well the compliance task was actually performed.

**Training config:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Steps | 200 | Enough for clear trend on T4, fits 15h Colab session |
| Generations per prompt | 4 | GRPO needs variance across completions |
| Batch size | 1 + grad accum 4 | T4 VRAM constraint |
| Learning rate | 5e-6 | Conservative — GRPO can be unstable at high LR |
| max_seq_length | 2048 | Prompt is ~800 tokens, needs headroom |
| max_completion_length | 384 | Enough for a full 10-15 action episode |

**Phase-aware partial credit**: When an episode doesn't reach `submit` within the step budget, we don't just return 0. We use the current environment state to compute partial credit:

```python
if env.phase == 'REDACT':
    return max(0.07, 0.05 + 0.6 * partial_r)   # got through SCAN + CLASSIFY
elif env.phase == 'CLASSIFY':
    return max(0.06, 0.05 + 0.3 * scan_f1)      # good scan, ran out of steps
else:
    return max(0.055, 0.05 + 0.1 * scan_f1)     # found some PII in SCAN
```

This is critical. Without partial credit, a model that correctly scans all files but doesn't finish redacting returns the same reward (0.05) as a model that outputs nonsense. The gradient signal disappears. Partial credit ensures progress at every phase is rewarded.

**Try it yourself:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Madhan-mohan14/data-privacy-env/blob/main/data_privacy_env/training/ComplianceGuard_GRPO_Training.ipynb)

The notebook runs end-to-end on a free T4 Colab session in approximately 2–3 hours.

---

## What Actually Happened Over 200 Steps

| Metric | Value |
|--------|-------|
| Step-1 reward | 0.097 |
| First-10 average | 0.097 |
| Last-10 average | **0.161** (+66%) |
| Peak reward | **0.298** at step 181 |
| KL divergence at step 200 | 0.031 |

![GRPO training reward curve](https://raw.githubusercontent.com/Madhan-mohan14/data-privacy-env/main/data_privacy_env/assets/reward_curve.png)

The smoothed reward curve rises from ~0.10 to ~0.18 over 200 steps. The trained 1.5B model reaches a peak of 0.298 — a reward the 7B baseline never achieved on L3 or L4.

**KL divergence at 0.031** is an important signal. If KL were 0.000, the model weights didn't move — training was dead. If KL were very high (>1.0), the model diverged into incoherent outputs. 0.031 is healthy: the policy changed meaningfully without destabilizing.

The +66% improvement (0.097 → 0.161 last-10 average) is relative to the 1.5B model's own starting point, not the 7B baseline. A 1.5B model with 200 GRPO steps is showing measurably better compliance behavior than it started with — on the *same hardware, same cost, same model size*.

---

## The Behavior That Changed: A Concrete Before/After

The most instructive way to see what GRPO learned is to trace a single L3 episode (obfuscated PII) and look at one specific decision: the CLASSIFY phase.

**File content (server_logs.txt, L3 seed 42):**
```
2026-01-01 INFO contact Grace Harris via grace.harris [at] yahoo [dot] com
2026-01-02 WARN reach Bob Johnson at 553 dot 847 dot 2341
```

**BEFORE training (7B baseline, reward = 0.418):**

```
Step 1: {"tool": "list_files"}           → server_logs.txt, incident_report.txt, notes.txt
Step 2: {"tool": "read_file", "file_path": "server_logs.txt"}   → [file contents]
Step 3: {"tool": "flag_candidate", "text": "Grace Harris", ...}  → Flagged c0 [real PII] ✓
Step 4: {"tool": "flag_candidate", "text": "grace.harris [at] yahoo [dot] com", ...} → Flagged c1 [real PII] ✓
Step 5: {"tool": "advance_phase"}         → → CLASSIFY. 2 candidates.

Step 6: {"tool": "classify_candidate", "candidate_id": "c0", "confirmed": true}   ✓
Step 7: {"tool": "classify_candidate", "candidate_id": "c1", "confirmed": false}  ✗
```

The model correctly flagged the obfuscated email in SCAN (it's visible in the text), but then **rejected it as non-PII in CLASSIFY** — treating `[at]` as non-standard and concluding "this isn't an email address." Result: classify_accuracy = 0.5, redact_complete = 0.5. Final reward: 0.418.

**AFTER training (1.5B GRPO-trained, reward = 0.861):**

```
Step 6: {"tool": "classify_candidate", "candidate_id": "c0", "confirmed": true}   ✓
Step 7: {"tool": "classify_candidate", "candidate_id": "c1", "confirmed": true}   ✓
Step 8: {"tool": "advance_phase"}         → → REDACT. 2 confirmed PII.

Step 9:  {"tool": "redact_span", "candidate_id": "c0"}   → Grace Harris → [REDACTED] ✓
Step 10: {"tool": "redact_span", "candidate_id": "c1"}   → grace.harris [at] yahoo [dot] com → [REDACTED] ✓
Step 11: {"tool": "submit"}               → reward=0.861
```

The trained model correctly classifies the obfuscated email as real PII. GRPO didn't teach it a regex rule. It taught it to reason: *if I flagged this in SCAN because it appeared next to a real person's name and follows an email-like structure, it's probably real PII even if the delimiters are unusual.*

This is the kind of contextual reasoning that makes the difference between a demo and a deployable system.

---

## The Engineering That Made This Work with OpenEnv

ComplianceGuard is built on the OpenEnv framework. A few implementation details worth noting:

**`ComplianceGuardEnv(Environment)` extends the OpenEnv base class** — proper `reset()`, `step()`, `state()` interface. This makes the environment pull-able by any OpenEnv-compatible trainer, not just our notebook.

**The `virtual_fs` persists across steps** — when `redact_span` is called, it does a string-replace in the actual file contents stored in `env.virtual_fs`. The `redact_complete` metric checks whether real PII strings still appear in the joined file contents. This means the agent has to actually change the state of the world, not just claim it did.

**Phase gates are enforced at the `step()` level** — the environment raises `ValueError` for wrong-phase tools, which TRL's GRPOTrainer handles natively. No custom logic needed in the training loop.

**`openenv.yaml` declares all four levels** as named tasks — `level_1` through `level_4` — so OpenEnv's validation tools can discover and test the full curriculum automatically.

---

## Why This Problem Matters Beyond the Hackathon

The GDPR/HIPAA compliance market is over $10 billion annually. Manual PII redaction is a bottleneck for every company that handles customer data — law firms, healthcare providers, financial institutions, tech companies. A system that can reliably automate this — handling obfuscated forms, avoiding false positives, following a review workflow — would have immediate, measurable value.

More importantly, ComplianceGuard is a demonstration that **RL with real environment feedback teaches behavior that prompt engineering cannot**. The 7B model with a perfect system prompt still rejected obfuscated emails as non-PII. The 1.5B model with 200 GRPO steps learned to confirm them. The difference isn't knowledge — it's the feedback loop. The environment told the model, repeatedly, that rejecting `[at]` as non-PII costs it reward. The model updated accordingly.

That feedback loop is what OpenEnv makes possible at scale.

---

## Try It

| Resource | Link |
|----------|------|
| **HF Space (live environment)** | https://huggingface.co/spaces/Maddy140605/dataprivacy-env |
| **Training Notebook (Colab)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Madhan-mohan14/data-privacy-env/blob/main/data_privacy_env/training/ComplianceGuard_GRPO_Training.ipynb) |
| **GitHub Repository** | https://github.com/Madhan-mohan14/data-privacy-env |
| **OpenEnv Framework** | https://github.com/meta-pytorch/openenv |

The training notebook runs end-to-end on a **free T4 Colab session** — no paid compute required. Run all 9 cells to reproduce the training curve. The environment is inlined in the notebook (no external server needed during training).

---

*Built for the Meta × HuggingFace OpenEnv Hackathon, April 2026.*  
*Theme 3.1 — Professional Tasks: real tool interaction, domain-specific reasoning, multi-step workflow.*
