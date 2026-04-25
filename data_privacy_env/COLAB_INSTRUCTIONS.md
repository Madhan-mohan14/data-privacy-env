# ComplianceGuard — Google Colab Training Instructions

Run GRPO training on a free T4 GPU using `training/ComplianceGuard_GRPO_Training.ipynb`.

## Quick Start

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `training/ComplianceGuard_GRPO_Training.ipynb`
3. Set runtime: **Runtime → Change runtime type → T4 GPU**
4. Run cells top-to-bottom

## Design: multi-action completions (no oracle)

Unsloth's patched GRPO removes `environment_factory` support, requiring `reward_funcs=`.  
`reward_funcs=` gives the model a single completion — so we ask the model to output
**ALL episode actions in one response** (one JSON per line).  
`compliance_reward` runs every action through the env and returns the **real episode reward**.

This is the critical difference from naive oracle-based approaches:
- **Broken (oracle hack):** model outputs 1 action → oracle completes episode → reward=0.999 always → reward_std=0.000 → model learns nothing
- **Fixed (this notebook):** model outputs ALL actions → run them for real → reward varies 0.05–1.0 → reward_std>0.05 → genuine learning

## Cell-by-Cell Guide

### Cell 1 — Install dependencies
```
!pip install unsloth trl>=0.15.0 datasets accelerate peft
```

### Cell 2 — ComplianceGuard environment (inline)
Full env class runs in-process — no server needed.

### Cell 3 — Reward function
```python
def compliance_reward(completions, prompts=None, **kwargs):
    ...
    for action in extract_json_actions(text):
        obs = env.step(action)
    return real_episode_reward  # no oracle
```
Reward health check: `reward_std` should be **> 0.05** after 5–10 steps.
If it's 0.000, the reward function is broken.

### Cell 4 — Dataset
Each row has `prompt` (system + initial obs), `level` (1–4), `seed`.
TRL passes `level` and `seed` as kwargs to `compliance_reward` so each
completion is evaluated against the correct episode.

### Cell 5 — Load model
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    'unsloth/Qwen2.5-1.5B-Instruct',
    max_seq_length=512,
    load_in_4bit=True,
    fast_inference=False,   # required on T4
)
```

### Cell 6 — Training
```python
config = GRPOConfig(
    output_dir='checkpoints/grpo',
    max_steps=200,
    num_generations=4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    use_vllm=False,            # required on T4
    max_completion_length=512, # needs to fit full action plan
    report_to='none',
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=config,               # use args=, NOT config=
    train_dataset=dataset,
    reward_funcs=compliance_reward,
)
trainer.train()
```

**Critical: use `args=config` not `config=config`** — Unsloth's patched GRPOTrainer changed the parameter name.

### Cell 7 — Save model
Saves adapter + merged 16-bit model.

### Cell 8 — Plot reward curves
Saves `reward_curve.png`.

### Cell 9 — Before/after comparison
Runs eval episodes with trained model.

### Cell 10 — Download results
```python
files.download('reward_log.csv')
files.download('reward_curve.png')
```

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `ImportError: Please install vLLM before enabling fast_inference` | Set `fast_inference=False` in model loading |
| `TypeError: got unexpected keyword argument 'config'` | Change `config=config` → `args=config` in GRPOTrainer |
| `AssertionError: environment_factory` | Unsloth removed environment_factory — use `reward_funcs=` (already fixed in this notebook) |
| `reward_std=0.000` at step 1 | Oracle hacking — this notebook's reward_funcs runs real episodes, so std should be >0 |
| `SyntaxError: unterminated f-string` | f-strings can't contain literal newlines; use `\n` escape inside them |
| CUDA OOM | Reduce `per_device_train_batch_size` to 1, `max_completion_length` to 256 |

## Expected Training Time

| GPU | Steps | Time |
|-----|-------|------|
| T4  | 200   | ~45–60 min |
| A100 | 200  | ~15–20 min |

## After Training

1. Download `reward_log.csv` and `reward_curve.png`
2. Place `reward_log.csv` in repo root
3. Run `python plot_baseline.py` to generate the baseline chart
4. Update README with results

## What healthy training looks like

```
step=1   reward=0.15  reward_std=0.12  ← real variance, not 0.000
step=10  reward=0.22  reward_std=0.18
step=50  reward=0.35  reward_std=0.21
step=100 reward=0.48  reward_std=0.24
step=200 reward=0.61  reward_std=0.19  ← model learning!
```

If you see `reward=0.999` from step 1 with `reward_std=0.000`, training is broken (oracle hacking).

## Environment Variables

```bash
export HF_TOKEN=hf_...         # Required for HF Inference API inference
export GROQ_API_KEY=gsk_...    # Alternative: Groq API
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct   # HF model override
```
