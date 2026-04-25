"""
GRPO training with Unsloth + TRL environment_factory.
Run on HF A100 compute (onsite Apr 25-26).

Smoke test locally (CPU, tiny model):
    python training/grpo_train.py --smoke-test
"""
import os
import argparse
import inspect

os.environ["CURRICULUM_TRAINING"] = "1"

# Unsloth must be patched BEFORE importing GRPOTrainer
from unsloth import FastLanguageModel, PatchFastRL  # noqa: E402

PatchFastRL("GRPO", FastLanguageModel)

from trl import GRPOConfig, GRPOTrainer  # noqa: E402
from datasets import Dataset  # noqa: E402

from training.grpo_env import ComplianceGuardEnvTRL  # noqa: E402
from agents.prompts import PHASE_PROMPTS  # noqa: E402

# Verify TRL supports environment_factory
assert "environment_factory" in inspect.signature(GRPOTrainer.__init__).parameters, (
    "TRL version does not support environment_factory. Run: pip install trl --upgrade"
)


def build_dataset(n_per_level: int = 100) -> Dataset:
    rows = []
    for level in range(1, 5):
        for seed in range(n_per_level):
            rows.append({
                "prompt": PHASE_PROMPTS["SCAN"],
                "level": level,
                "seed": seed + level * 1000,
            })
    return Dataset.from_list(rows)


def main(smoke: bool = False):
    model_name = "unsloth/Qwen3-0.6B" if smoke else "unsloth/Qwen3-1.7B"
    max_steps = 5 if smoke else 200

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=not smoke,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",
    )

    config = GRPOConfig(
        output_dir="checkpoints/grpo",
        max_steps=max_steps,
        num_generations=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        use_vllm=not smoke,
        vllm_mode="colocate" if not smoke else None,
        logging_steps=1,
        save_steps=50,
        report_to="none" if smoke else "wandb",
    )

    dataset = build_dataset(n_per_level=5 if smoke else 100)

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=dataset,
        environment_factory=ComplianceGuardEnvTRL,
    )
    trainer.train()

    if not smoke:
        model.save_pretrained("checkpoints/grpo/final")
        print("Training complete. Checkpoint saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    main(smoke=args.smoke_test)
