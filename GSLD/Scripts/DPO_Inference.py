"""
This script performs inference for a DPO-trained legal text simplification model.
It loads a base causal language model together with a PEFT (LoRA/DPO) adapter and
generates simplified versions of legal paragraphs at different proficiency levels.

The system supports three graded simplification settings:
- Skilled: Preserves legal terminology while improving clarity and structure.
- Intermediate: Simplifies vocabulary and sentence structure for semi-expert readers.
- Basic: Produces highly simplified output for non-legal audiences.

Input:
- A JSONL file containing original legal text along with preferred and non-preferred
  reference simplifications used during DPO training.

Output:
- CSV and Excel files containing the original text, human-preferred reference output,
  and the model-generated simplification.
"""


import argparse
import json
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig
from huggingface_hub import login

# =========================
# Grade → Instruction Mapping
# =========================
GRADE_TO_INSTRUCTION = {
    "skilled": (
        "You are a legal expert responsible for simplifying a legal paragraph. "
        "Rewrite the text to improve clarity and readability while preserving "
        "legal accuracy and domain-specific terminology. "
        "Only output the simplified text, nothing else:"
    ),
    "intermediate": (
        "You are a legal expert responsible for simplifying a legal paragraph. "
        "Rewrite complex legal vocabulary into simpler equivalents and break down "
        "long sentences, making the content accessible to readers with limited "
        "legal knowledge while preserving the original meaning. "
        "Only output the simplified text, nothing else:"
    ),
    "basic": (
        "You are a legal expert simplifying a legal paragraph for a general audience "
        "with no legal background. Replace complex legal terms with very simple words "
        "and short sentences while keeping the original meaning. "
        "Only output the simplified text, nothing else:"
    ),
}

# =========================
# Dataset Loader
# =========================
def build_dataset(path):
    originals, chosen, rejected = [], [], []

    with open(path, "r") as infile:
        for line in infile:
            ex = json.loads(line)
            originals.append(ex["original"])
            chosen.append(ex["preferred"])
            rejected.append(ex["not-preferred"])

    df = pd.DataFrame({
        "prompt": originals,
        "chosen": chosen,
        "rejected": rejected,
    })

    return Dataset.from_pandas(df)

# =========================
# Model & Tokenizer Loader
# =========================
def load_model_and_tokenizer(model_name, peft_path):
    tokenizer = AutoTokenizer.from_pretrained(peft_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )

    model = PeftModel.from_pretrained(
        base_model,
        peft_path,
        is_trainable=False,
    )

    model.eval()
    return model, tokenizer

# =========================
# Inference Function
# =========================
def run_inference(args):
    # Authenticate
    login(token=args.hf_token)

    instruction = GRADE_TO_INSTRUCTION[args.grade.lower()]

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.peft_path,
    )

    dataset = build_dataset(args.input_file)
    results = []

    for idx in tqdm(range(len(dataset))):
        prompt = instruction + " " + dataset[idx]["prompt"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.6,
                top_p=0.9,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        # Strip prompt from output
        if decoded.startswith(prompt):
            simplified = decoded[len(prompt):].strip()
        else:
            simplified = decoded.strip()

        results.append({
            "original": dataset[idx]["prompt"],
            "preferred_response": dataset[idx]["chosen"],
            "generated_response": simplified,
        })

    return results

# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference for DPO-based graded legal simplification"
    )

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--peft_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument(
        "--grade",
        type=str,
        choices=["skilled", "intermediate", "basic"],
        required=True,
    )
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="dpo_inference",
        help="Prefix for output CSV/XLSX files",
    )

    args = parser.parse_args()

    outputs = run_inference(args)

    df = pd.DataFrame(outputs)
    df.to_csv(f"{args.output_prefix}.csv", index=False)
    df.to_excel(f"{args.output_prefix}.xlsx", index=False)
