#!/usr/bin/env python3
"""
Paired error analysis for two adapters on MMLU using the same choice-scoring
logic as the strict full-eval scripts.
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_JORA = "/tmp/jora-stage3/out_jora_sweep6"
DEFAULT_LORA = "/tmp/jora-stage7/out_lora_r4"
DEFAULT_OUTPUT_DIR = "/tmp/jora-stage3/paired_analysis"
DEFAULT_LOW_SAMPLE_THRESHOLD = 120
CHOICE_TEXTS = [" A", " B", " C", " D"]


def load_adapter_config(adapter_path):
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing adapter_config.json: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def detect_model_id(jora_path, lora_path, override_model_id=None):
    if override_model_id:
        return override_model_id

    candidate_ids = {
        load_adapter_config(jora_path).get("base_model_name_or_path"),
        load_adapter_config(lora_path).get("base_model_name_or_path"),
    }
    candidate_ids.discard(None)

    if not candidate_ids:
        raise ValueError("Could not detect base model from adapter_config.json")
    if len(candidate_ids) != 1:
        raise ValueError(f"Adapter base models disagree: {sorted(candidate_ids)}")
    return next(iter(candidate_ids))


def load_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_adapter(model_id, adapter_path):
    print(f"Loading base model: {model_id}")
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model


def format_prompt(row):
    question = row["question"]
    choices = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(row["choices"])])
    return f"{question}\n{choices}\nAnswer:"


def score_choice(model, tokenizer, prompt, choice_text):
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(model.device)
    full_ids = tokenizer(prompt + choice_text, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(model.device)
    choice_ids = full_ids[:, prompt_ids.shape[1]:]
    if choice_ids.numel() == 0:
        return float("-inf")

    with torch.no_grad():
        outputs = model(full_ids)
        log_probs = outputs.logits[:, :-1, :].log_softmax(dim=-1)

    start = prompt_ids.shape[1] - 1
    end = full_ids.shape[1] - 1
    token_log_probs = log_probs[:, start:end, :]
    gathered = token_log_probs.gather(-1, choice_ids.unsqueeze(-1)).squeeze(-1)
    return gathered.sum().item()


def predict(model, tokenizer, prompt):
    choice_scores = [score_choice(model, tokenizer, prompt, choice_text) for choice_text in CHOICE_TEXTS]
    return max(range(len(choice_scores)), key=lambda idx: choice_scores[idx])


def get_predictions(adapter_path, label, model_id, tokenizer, records):
    model = load_model_and_adapter(model_id, adapter_path)
    predictions = []
    correct = 0

    print(f"Running predictions for {label}...")
    for record in tqdm(records, desc=label):
        pred_idx = predict(model, tokenizer, record["prompt"])
        predictions.append(pred_idx)
        if pred_idx == record["true_idx"]:
            correct += 1

    del model
    torch.cuda.empty_cache()

    accuracy = correct / len(records)
    print(f"{label}: {accuracy:.4f} ({correct}/{len(records)})")
    return predictions, correct


def build_subject_stats(results, low_sample_threshold):
    subject_stats = {}
    for result in results:
        subject = result["subject"]
        stats = subject_stats.setdefault(
            subject,
            {
                "total": 0,
                "jora_correct": 0,
                "lora_correct": 0,
                "both_correct": 0,
                "both_wrong": 0,
                "jora_better": 0,
                "lora_better": 0,
            },
        )

        stats["total"] += 1
        stats["jora_correct"] += int(result["jora_correct"])
        stats["lora_correct"] += int(result["lora_correct"])
        if result["jora_correct"] and result["lora_correct"]:
            stats["both_correct"] += 1
        elif not result["jora_correct"] and not result["lora_correct"]:
            stats["both_wrong"] += 1
        elif result["jora_correct"]:
            stats["jora_better"] += 1
        else:
            stats["lora_better"] += 1

    sorted_subjects = []
    for subject, stats in subject_stats.items():
        total = stats["total"]
        stats["jora_acc"] = stats["jora_correct"] / total
        stats["lora_acc"] = stats["lora_correct"] / total
        stats["accuracy_delta"] = stats["jora_acc"] - stats["lora_acc"]
        stats["net_delta"] = stats["jora_better"] - stats["lora_better"]
        stats["caution_low_sample"] = total <= low_sample_threshold

        sorted_subjects.append(
            {
                "subject": subject,
                **stats,
            }
        )

    sorted_subjects.sort(key=lambda item: (-item["net_delta"], -item["total"], item["subject"]))
    return subject_stats, sorted_subjects


def validate_counts(output_data):
    paired = output_data["paired_comparison"]
    total = output_data["num_samples"]
    jora_correct = output_data["overall"]["jora_correct"]
    lora_correct = output_data["overall"]["lora_correct"]

    if paired["both_correct"] + paired["both_wrong"] + paired["jora_better"] + paired["lora_better"] != total:
        raise ValueError("Paired counts do not sum to total samples")
    if paired["both_correct"] + paired["jora_better"] != jora_correct:
        raise ValueError("JORA correct count does not match paired breakdown")
    if paired["both_correct"] + paired["lora_better"] != lora_correct:
        raise ValueError("LoRA correct count does not match paired breakdown")

    subject_total = sum(item["total"] for item in output_data["subject_stats"].values())
    subject_net = sum(item["net_delta"] for item in output_data["subjects_sorted_by_net_delta"])
    if subject_total != total:
        raise ValueError("Subject totals do not sum to total samples")
    if subject_net != paired["jora_better"] - paired["lora_better"]:
        raise ValueError("Subject net deltas do not sum to overall net delta")


def run_paired_analysis(
    jora_path,
    lora_path,
    num_samples=0,
    output_name="paired",
    model_id=None,
    low_sample_threshold=DEFAULT_LOW_SAMPLE_THRESHOLD,
):
    resolved_model_id = detect_model_id(jora_path, lora_path, model_id)
    print(f"Resolved base model: {resolved_model_id}")

    tokenizer = load_tokenizer(resolved_model_id)

    ds = load_dataset("cais/mmlu", "all", split="test")
    print(f"MMLU test samples: {len(ds)}")
    if num_samples > 0:
        ds = ds.select(range(num_samples))
        print(f"Using first {num_samples} samples")

    records = []
    for idx, row in enumerate(ds):
        records.append(
            {
                "idx": idx,
                "subject": row.get("subject", "unknown"),
                "question": row["question"][:100],
                "prompt": format_prompt(row),
                "true_idx": int(row["answer"]),
            }
        )

    jora_predictions, jora_correct = get_predictions(jora_path, "JORA", resolved_model_id, tokenizer, records)
    lora_predictions, lora_correct = get_predictions(lora_path, "LoRA", resolved_model_id, tokenizer, records)

    results = []
    for record, jora_pred, lora_pred in zip(records, jora_predictions, lora_predictions):
        jora_is_correct = jora_pred == record["true_idx"]
        lora_is_correct = lora_pred == record["true_idx"]
        results.append(
            {
                "idx": record["idx"],
                "subject": record["subject"],
                "question": record["question"],
                "true_idx": record["true_idx"],
                "jora_pred": jora_pred,
                "lora_pred": lora_pred,
                "jora_correct": jora_is_correct,
                "lora_correct": lora_is_correct,
            }
        )

    total = len(results)
    jora_acc = jora_correct / total
    lora_acc = lora_correct / total
    both_correct = sum(1 for item in results if item["jora_correct"] and item["lora_correct"])
    both_wrong = sum(1 for item in results if not item["jora_correct"] and not item["lora_correct"])
    jora_better = sum(1 for item in results if item["jora_correct"] and not item["lora_correct"])
    lora_better = sum(1 for item in results if not item["jora_correct"] and item["lora_correct"])

    print("\n=== Overall Results ===")
    print(f"JORA: {jora_acc:.4f} ({jora_correct}/{total})")
    print(f"LoRA: {lora_acc:.4f} ({lora_correct}/{total})")
    print(f"JORA minus LoRA: {jora_acc - lora_acc:+.4f}")

    print("\n=== Paired Comparison ===")
    print(f"Both correct: {both_correct}")
    print(f"Both wrong: {both_wrong}")
    print(f"JORA better: {jora_better}")
    print(f"LoRA better: {lora_better}")
    print(f"Net delta: {jora_better - lora_better:+d}")

    subject_stats, sorted_subjects = build_subject_stats(results, low_sample_threshold)

    print("\n=== Subject Stats (sorted by net delta) ===")
    print(f"{'Subject':<40} {'Net':>6} {'Total':>7} {'JORA>LoRA':>10} {'LoRA>JORA':>10} {'Caution':>8}")
    print("-" * 90)
    for item in sorted_subjects:
        caution = "yes" if item["caution_low_sample"] else "no"
        print(
            f"{item['subject']:<40} {item['net_delta']:>+6d} {item['total']:>7d} "
            f"{item['jora_better']:>10d} {item['lora_better']:>10d} {caution:>8}"
        )

    output_file = os.path.join(
        DEFAULT_OUTPUT_DIR,
        output_name if output_name.endswith(".json") else f"{output_name}.json",
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    output_data = {
        "metadata": {
            "model_id": resolved_model_id,
            "jora_path": jora_path,
            "lora_path": lora_path,
            "low_sample_threshold": low_sample_threshold,
        },
        "num_samples": total,
        "overall": {
            "jora_acc": jora_acc,
            "lora_acc": lora_acc,
            "jora_correct": jora_correct,
            "lora_correct": lora_correct,
            "jora_minus_lora": jora_acc - lora_acc,
        },
        "paired_comparison": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "jora_better": jora_better,
            "lora_better": lora_better,
            "net_delta": jora_better - lora_better,
        },
        "subject_stats": subject_stats,
        "subjects_sorted_by_net_delta": sorted_subjects,
        "detailed_results": results,
    }

    validate_counts(output_data)

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved detailed results to {output_file}")

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jora", default=DEFAULT_JORA, help="JORA adapter path")
    parser.add_argument("--lora", default=DEFAULT_LORA, help="LoRA adapter path")
    parser.add_argument("--samples", type=int, default=0, help="Number of samples (0=full)")
    parser.add_argument("--name", default="stage3_vs_stage7", help="Output name")
    parser.add_argument("--model-id", default=None, help="Override base model id")
    parser.add_argument(
        "--low-sample-threshold",
        type=int,
        default=DEFAULT_LOW_SAMPLE_THRESHOLD,
        help="Mark subjects with total <= threshold as low-sample caution",
    )
    args = parser.parse_args()

    run_paired_analysis(
        args.jora,
        args.lora,
        args.samples,
        args.name,
        args.model_id,
        args.low_sample_threshold,
    )
