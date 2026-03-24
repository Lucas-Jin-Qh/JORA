#!/usr/bin/env bash
# Stage 1.1: Paired comparison analysis (fixed NUM_SAMPLES)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/jqh/miniconda3/envs/peft-jora/bin/python}"
export PYTHONPATH="${REPO_ROOT}/src"

MODEL_ID="${MODEL_ID:-facebook/opt-1.3b}"
WORKDIR="${WORKDIR:-/tmp/jora-stage7}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"  # Default to 100, can be overridden

echo "=== Stage 1.1: Paired Comparison Analysis ==="
echo "Model: $MODEL_ID"
echo "Samples: $NUM_SAMPLES"
echo ""

mkdir -p "$WORKDIR/paired_analysis"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" - <<'PYEOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from collections import defaultdict
from datasets import load_dataset
import os

MODEL_ID = os.environ.get("MODEL_ID", "facebook/opt-1.3b")
WORKDIR = os.environ.get("WORKDIR", "/tmp/jora-stage7")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "100"))
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", f"{WORKDIR}/paired_analysis/results_quick.json")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Load MMLU once
print("Loading MMLU dataset...")
ds = load_dataset("cais/mmlu", "all", split="test")
ds = ds.select(range(NUM_SAMPLES))
print(f"Using {NUM_SAMPLES} samples")

def format_prompt(row):
    question = row["question"]
    choices = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(row["choices"])])
    return f"{question}\n{choices}\nAnswer:"

choice_texts = [" A", " B", " C", " D"]

# Pre-compute all prompts
print("Pre-computing prompts...")
prompts = [format_prompt(row) for row in ds]
answers = [row["answer"] for row in ds]
subjects = [row.get("subject", "unknown") for row in ds]

def get_predictions(model, model_name):
    """Get predictions for all prompts"""
    print(f"\nRunning predictions with {model_name}...")
    all_choice_scores = []
    for choice_text in choice_texts:
        scores = []
        for prompt in prompts:
            prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(model.device)
            full_ids = tokenizer(prompt + choice_text, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(model.device)
            choice_ids = full_ids[:, prompt_ids.shape[1]:]
            if choice_ids.numel() == 0:
                scores.append(float("-inf"))
                continue

            with torch.no_grad():
                outputs = model(full_ids)
                log_probs = outputs.logits[:, :-1, :].log_softmax(dim=-1)

            start = prompt_ids.shape[1] - 1
            end = full_ids.shape[1] - 1
            token_log_probs = log_probs[:, start:end, :]
            gathered = token_log_probs.gather(-1, choice_ids.unsqueeze(-1)).squeeze(-1)
            scores.append(gathered.sum().item())
        all_choice_scores.append(scores)
    
    # Get predictions
    predictions = []
    for i in range(len(prompts)):
        choice_scores = [all_choice_scores[c][i] for c in range(len(choice_texts))]
        pred_idx = max(range(len(choice_scores)), key=lambda idx: choice_scores[idx])
        predictions.append(pred_idx)
    
    # Calculate accuracy
    correct = sum(1 for i, pred in enumerate(predictions) if pred == answers[i])
    accuracy = correct / len(predictions)
    print(f"  {model_name} Accuracy: {accuracy:.4f} ({correct}/{len(predictions)})")
    
    return predictions

# List of models to evaluate
models_to_eval = [
    (MODEL_ID, "base", False),
    (f"{WORKDIR}/out_jora_oer", "jora_oer", True),
    (f"{WORKDIR}/out_jora_none", "jora_none", True),
    (f"{WORKDIR}/out_lora_r4", "lora", True),
]

all_predictions = {}

for model_path, key_name, is_adapter in models_to_eval:
    print(f"\nLoading {key_name}...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    
    if is_adapter:
        adapter_config = os.path.join(model_path, "adapter_config.json")
        if not os.path.exists(adapter_config):
            raise FileNotFoundError(f"Missing adapter_config.json: {adapter_config}")
        model = PeftModel.from_pretrained(base, model_path)
        print(f"  Loaded adapter from {model_path}")
    else:
        model = base
        print(f"  Using base model")
    
    model.eval()
    all_predictions[key_name] = get_predictions(model, key_name)
    
    del model
    torch.cuda.empty_cache()

# Analysis
print("\n" + "="*60)
print(f"PAIRED COMPARISON ANALYSIS ({NUM_SAMPLES} samples)")
print("="*60)

paired = []
for i in range(len(prompts)):
    paired.append({
        "idx": i,
        "subject": subjects[i],
        "base_correct": all_predictions["base"][i] == answers[i],
        "jora_oer_correct": all_predictions["jora_oer"][i] == answers[i],
        "jora_none_correct": all_predictions["jora_none"][i] == answers[i],
        "lora_correct": all_predictions["lora"][i] == answers[i],
    })

# Overall stats
base_acc = sum(p["base_correct"] for p in paired) / len(paired)
jora_oer_acc = sum(p["jora_oer_correct"] for p in paired) / len(paired)
jora_none_acc = sum(p["jora_none_correct"] for p in paired) / len(paired)
lora_acc = sum(p["lora_correct"] for p in paired) / len(paired)

print(f"\nOverall Accuracy:")
print(f"  Base:       {base_acc:.4f}")
print(f"  JORA-oer:   {jora_oer_acc:.4f} (Δ={jora_oer_acc - base_acc:+.4f})")
print(f"  JORA-none:  {jora_none_acc:.4f} (Δ={jora_none_acc - base_acc:+.4f})")
print(f"  LoRA-r4:    {lora_acc:.4f} (Δ={lora_acc - base_acc:+.4f})")

# JORA-oer vs LoRA comparison
print(f"\nJORA-oer vs LoRA-r4:")
jora_oer_better = sum(1 for p in paired if p["jora_oer_correct"] and not p["lora_correct"])
jora_oer_worse = sum(1 for p in paired if not p["jora_oer_correct"] and p["lora_correct"])
both_correct = sum(1 for p in paired if p["jora_oer_correct"] and p["lora_correct"])
both_wrong = sum(1 for p in paired if not p["jora_oer_correct"] and not p["lora_correct"])

print(f"  JORA better than LoRA:  {jora_oer_better}")
print(f"  JORA worse than LoRA:    {jora_oer_worse}")
print(f"  Both correct:            {both_correct}")
print(f"  Both wrong:              {both_wrong}")
print(f"  Net difference:          {jora_oer_better - jora_oer_worse}")

# JORA-none vs LoRA comparison
print(f"\nJORA-none vs LoRA-r4:")
jora_none_better = sum(1 for p in paired if p["jora_none_correct"] and not p["lora_correct"])
jora_none_worse = sum(1 for p in paired if not p["jora_none_correct"] and p["lora_correct"])
both_correct_none = sum(1 for p in paired if p["jora_none_correct"] and p["lora_correct"])
both_wrong_none = sum(1 for p in paired if not p["jora_none_correct"] and not p["lora_correct"])

print(f"  JORA better than LoRA:  {jora_none_better}")
print(f"  JORA worse than LoRA:    {jora_none_worse}")
print(f"  Both correct:            {both_correct_none}")
print(f"  Both wrong:              {both_wrong_none}")
print(f"  Net difference:          {jora_none_better - jora_none_worse}")

# Subject-level analysis
print(f"\n" + "="*60)
print("BY SUBJECT (JORA-oer vs LoRA)")
print("="*60)

subject_stats = defaultdict(lambda: {"jora_better": 0, "lora_better": 0, "both_correct": 0, "both_wrong": 0, "total": 0})

for p in paired:
    subj = p["subject"]
    subject_stats[subj]["total"] += 1
    if p["jora_oer_correct"] and not p["lora_correct"]:
        subject_stats[subj]["jora_better"] += 1
    elif not p["jora_oer_correct"] and p["lora_correct"]:
        subject_stats[subj]["lora_better"] += 1
    elif p["jora_oer_correct"] and p["lora_correct"]:
        subject_stats[subj]["both_correct"] += 1
    else:
        subject_stats[subj]["both_wrong"] += 1

# Sort by absolute difference
sorted_subjects = sorted(
    subject_stats.items(),
    key=lambda x: abs(x[1]["jora_better"] - x[1]["lora_better"]),
    reverse=True
)

print(f"\n{'Subject':<40} {'JORA>LoRA':>10} {'LoRA>JORA':>10}")
print("-" * 60)
for subj, stats in sorted_subjects[:20]:
    net = stats["jora_better"] - stats["lora_better"]
    print(f"{subj:<40} {stats['jora_better']:>10} {stats['lora_better']:>10}")

# Save detailed results
with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "num_samples": NUM_SAMPLES,
        "overall": {
            "base_acc": base_acc,
            "jora_oer_acc": jora_oer_acc,
            "jora_none_acc": jora_none_acc,
            "lora_acc": lora_acc,
        },
        "paired_comparison": {
            "jora_oer_vs_lora": {
                "jora_better": jora_oer_better,
                "lora_better": jora_oer_worse,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
            },
            "jora_none_vs_lora": {
                "jora_better": jora_none_better,
                "lora_better": jora_none_worse,
                "both_correct": both_correct_none,
                "both_wrong": both_wrong_none,
            }
        },
        "subject_stats": dict(subject_stats),
    }, f, indent=2)

print(f"\nResults saved to: {OUTPUT_FILE}")
PYEOF
