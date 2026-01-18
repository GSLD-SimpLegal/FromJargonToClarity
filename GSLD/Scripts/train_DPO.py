import json
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig

# === Step 1: Load JSON Dataset ===

def load_json_dataset(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return Dataset.from_list(data)

# === Step 2: Collator ===

def build_collate_fn(tokenizer, max_length=512):
    def collate_fn(batch):
        chosen = tokenizer(
            [b["prompt"] + " " + b["chosen"] for b in batch],
            truncation=True,
            max_length=max_length,
            padding="longest",
            return_tensors="pt",
        )
        rejected = tokenizer(
            [b["prompt"] + " " + b["rejected"] for b in batch],
            truncation=True,
            max_length=max_length,
            padding="longest",
            return_tensors="pt",
        )
        return {
            "input_ids_chosen": chosen["input_ids"],
            "attention_mask_chosen": chosen["attention_mask"],
            "input_ids_rejected": rejected["input_ids"],
            "attention_mask_rejected": rejected["attention_mask"],
        }
    return collate_fn

# === Step 3: Load Model + Tokenizer with Quantization ===

def load_model_and_tokenizer(model_name, quant_bits):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if quant_bits == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )
    elif quant_bits == 8:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype="auto",
    )

    return model, tokenizer

# === Step 4: Training ===

def train_dpo(args):
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        args.quant_bits
    )

    dataset = load_json_dataset(args.data_path)

    dpo_config = DPOConfig(
        beta=0.1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        output_dir=args.output_dir,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=build_collate_fn(tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)

# === CLI ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO training with quantization")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name or local path"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to JSON preference dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dpo-output",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--quant_bits",
        type=int,
        choices=[4, 8, 16],
        default=4,
        help="Quantization bits (4, 8, or 16=no quant)"
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=3)

    args = parser.parse_args()
    train_dpo(args)
