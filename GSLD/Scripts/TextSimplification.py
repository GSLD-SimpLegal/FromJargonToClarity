import numpy as np
import pandas as pd
import json
import time, datetime
import re
import argparse
import torch
from os import listdir, path
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig
from huggingface_hub import login

# ======================================================
# Utility helpers
# ======================================================

def print_curr_time():
    print(datetime.datetime.now())

def write_jsonl(out_filename, list_of_dicts):
    with open(out_filename, "w", encoding="utf-8") as f:
        for row in list_of_dicts:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

def extract_simplified_response(text):
    match = re.search(r"<simplified_response>(.*?)</simplified_response>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

# ======================================================
# IGIP Instruction Dictionary
# ======================================================

IGIP_INSTRUCTIONS = {
    "zs": {
        "G1": """You are a legal expert simplifying a legal paragraph. Replace complex legal vocabulary with simpler equivalents and improve sentence clarity while preserving meaning. Provide only the simplified text.""",

        "G2": """You are a legal expert simplifying a legal paragraph further for a general audience. Improve clarity and accessibility while preserving intent. Provide only the simplified text.""",

        "G3": """You are a legal expert simplifying a legal paragraph for readers with no legal background. Use very simple words and short sentences while preserving meaning. Provide only the simplified text."""
    },

    "igip": {
        "G1": """You are a legal expert simplifying a legal paragraph while preserving legal accuracy. Improve structure and clarity while maintaining terminology. Target a Flesch Reading Ease of 60–80. Provide only one simplified version.""",

        "G2": """You are a legal expert simplifying a legal paragraph using iterative refinement. Improve clarity and accessibility while preserving intent. Target a Flesch Reading Ease of 70–80. Provide only one simplified version.""",

        "G3": """You are a legal expert simplifying a legal paragraph for a very general audience. Use simple words and short sentences. Target a Flesch Reading Ease of 90–100. Provide only one simplified version."""
    }
}

# ======================================================
# Core generation functions
# ======================================================

def process_paragraph(paragraph, memory, instruction, tokenizer, model, use_context, max_tokens=256):
    """
    Generates a simplified paragraph with or without document memory.
    """
    if use_context:
        prompt = f"""
You are a responsible legal AI.

Task:
{instruction}

legal_text:
{paragraph}

legal_doc:
{memory}

Simplified Output:
"""
    else:
        prompt = f"""
You are a responsible legal AI.

Task:
{instruction}

legal_text:
{paragraph}

Simplified Output:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9
        )

    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# ======================================================
# Iterative IGIP variants
# ======================================================

def fixed_igip_iterations(doc_data, instruction, tokenizer, model, use_context):
    results = []
    for entry in tqdm(doc_data):
        text = entry["Original"]
        memory = entry.get("Memory", "")
        if len(text.split()) < 50:
            continue

        outputs = {"Original": text}
        for i in range(10):
            outputs[str(i)] = process_paragraph(
                text, memory, instruction, tokenizer, model, use_context
            )
        results.append(outputs)

        if len(results) == 500:
            break
    return results

def single_pass(doc_data, instruction, tokenizer, model, use_context):
    results = []
    for entry in tqdm(doc_data):
        text = entry["Original"]
        memory = entry.get("Memory", "")
        if len(text.split()) < 50:
            continue

        simplified = process_paragraph(
            text, memory, instruction, tokenizer, model, use_context
        )
        results.append({"Original": text, "Simplified": simplified})
    return results

# ======================================================
# Main
# ======================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_token", required=True)
    parser.add_argument("--model_path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--method", choices=["zs", "igip"], default="zs")
    parser.add_argument("--granularity", choices=["G1", "G2", "G3"], default="G1")
    parser.add_argument("--context", choices=["yes", "no"], default="no")
    parser.add_argument("--memory_file", required=True)
    parser.add_argument("--output_dir", default="Outputs/SimplifiedEnglish2")

    args = parser.parse_args()

    login(token=args.hf_token)

    instruction = IGIP_INSTRUCTIONS[args.method][args.granularity]
    use_context = args.context == "yes"

    with open(args.memory_file, "r") as f:
        doc_data = [json.loads(line) for line in f]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.config.use_cache = False

    if args.method == "igip":
        outputs = fixed_igip_iterations(doc_data, instruction, tokenizer, model, use_context)
        df = pd.DataFrame(outputs)
    else:
        outputs = single_pass(doc_data, instruction, tokenizer, model, use_context)
        df = pd.DataFrame(outputs)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = f"{args.output_dir}/{args.method}_{args.granularity}_{args.context}.xlsx"
    df.to_excel(out_path, index=False)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    print_curr_time()
    main()
    print_curr_time()
