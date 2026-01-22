"""
This script trains a graded legal text simplification model using Direct Preference
Optimization (DPO). It fine-tunes a quantized causal language model with LoRA adapters
to align generation behavior with human preference data.

The training setup supports three simplification proficiency levels:
- Skilled: Improves readability while preserving formal legal terminology.
- Intermediate: Simplifies vocabulary and sentence structure for semi-expert readers.
- Basic: Produces highly simplified output for non-legal audiences.

Key Components:
- Preference-based training using (prompt, chosen, rejected) triplets.
- Parameter-efficient fine-tuning with LoRA adapters.
- Memory-efficient 4-bit quantized model loading.
- Automatic dataset filtering based on token length constraints.
- HuggingFace TRL DPOTrainer for stable preference optimization.

Input:
- A JSONL dataset containing original legal text, preferred simplifications, and
  non-preferred alternatives.

Output:
- A trained PEFT adapter checkpoint.
- Automatically pushed model and tokenizer artifacts to the HuggingFace Hub.

This script is intended for training controllable legal simplification models using
human-aligned preference learning and is designed for scalable experimentation across
multiple simplification grades.
"""
import os
import json
import argparse
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from huggingface_hub import login
from trl import DPOConfig, DPOTrainer

# =========================
# Global configuration
# =========================
max_seq_length = 1024
max_total_length = 2048

# =========================
# Grade → Instruction Mapping
# =========================
GRADE_TO_INSTRUCTION = {
    "skilled": (
        "You are a legal expert responsible for simplifying a legal paragraph. "
        "Rewrite the text to improve clarity and readability while preserving "
        "legal accuracy and domain-specific terminology."
    ),
    "intermediate": (
        "You are a legal expert responsible for simplifying a legal paragraph. "
        "Rewrite the key ideas using simpler language and shorter sentences, "
        "making the content accessible to readers with limited legal knowledge "
        "while preserving the original meaning."
    ),
    "basic": (
        "You are a legal expert responsible for simplifying a legal paragraph. "
        "Rewrite the content using very simple language and short sentences so "
        "that it is understandable to a general audience with no legal background, "
        "while preserving the original intent."
    ),
}

# =========================
# Dataset Construction
# =========================
def build_dataset(path, tokenizer, instruction):
    original_text, accepted_list, rejected_list = [], [], []

    with open(path, "r") as infile:
        for line in infile:
            example = json.loads(line)
            original_text.append(example["original"])
            accepted_list.append(example["preferred"])
            rejected_list.append(example["not-preferred"])

    df = pd.DataFrame({
        "prompt": original_text,
        "chosen": accepted_list,
        "rejected": rejected_list,
    }).sample(frac=1).reset_index(drop=True)

    dataset = Dataset.from_pandas(df)
    split_idx = int(0.8 * len(dataset))

    def _concat_length(prompt, text):
        full_prompt = instruction + " " + prompt
        combined = full_prompt + tokenizer.eos_token + text.strip()
        return len(tokenizer.encode(combined))

    def _filter_long_examples(example):
        return (
            _concat_length(example["prompt"], example["chosen"]) <= max_seq_length
            and _concat_length(example["prompt"], example["rejected"]) <= max_seq_length
        )

    dataset = dataset.filter(_filter_long_examples, num_proc=4)

    train_dataset = Dataset.from_pandas(dataset.to_pandas()[:split_idx])
    eval_dataset = Dataset.from_pandas(dataset.to_pandas()[split_idx:])

    return train_dataset, eval_dataset

# =========================
# Model & Tokenizer Loading
# =========================
def load_model_and_tokenizer(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    return model, tokenizer

# =========================
# Training Function
# =========================
def train_dpo_model(args):
    # Authenticate with Hugging Face
    login(token=args.hf_token)

    instruction = GRADE_TO_INSTRUCTION[args.grade.lower()]

    model, tokenizer = load_model_and_tokenizer(args.model)

    train_dataset, eval_dataset = build_dataset(
        args.input_file, tokenizer, instruction
    )

    run_name = f"{args.model.split('/')[-1]}-{args.grade}-DPO"
    output_dir = f"./{run_name}"

    dpo_config = DPOConfig(
        beta=0.1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        bf16=True,
        learning_rate=2.5e-5,
        num_train_epochs=1,
        logging_steps=1000,
        do_eval=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        output_dir=output_dir,
        remove_unused_columns=False,
        push_to_hub=True,
        gradient_checkpointing=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    trainer.push_to_hub(run_name)
    tokenizer.push_to_hub(run_name)

# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DPO training for graded legal simplification"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to JSONL DPO dataset",
    )
    parser.add_argument(
        "--grade",
        type=str,
        choices=["skilled", "intermediate", "basic"],
        required=True,
        help="Target simplification grade",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="Hugging Face access token",
    )

    args = parser.parse_args()
    train_dpo_model(args)
