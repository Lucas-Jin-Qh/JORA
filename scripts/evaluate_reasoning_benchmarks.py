#!/usr/bin/env python3
"""Evaluate a base model or adapter on MMLU, ARC-Challenge, and GSM8K.

The evaluator is intentionally simple and single-GPU friendly:

- MMLU and ARC-Challenge use per-choice log-likelihood scoring on answer labels.
- GSM8K uses deterministic generation and numeric exact match on the final answer.
- When loading an adapter, the model is loaded with `is_trainable=True` so
  reported PEFT trainable-parameter counts match the adapter's actual budget.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel


NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
BENCHMARK_CHOICES = ("mmlu", "arc_challenge", "gsm8k")


def _torch_dtype_from_name(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def _canonicalize_number(value: str) -> str:
    text = value.strip().replace(",", "")
    if not text:
        return ""
    try:
        decimal_value = Decimal(text)
    except InvalidOperation:
        return text
    normalized = format(decimal_value.normalize(), "f").rstrip("0").rstrip(".")
    if normalized in {"", "-0"}:
        return "0"
    return normalized


def extract_final_numeric_answer(text: str) -> str:
    candidate = text.split("####")[-1] if "####" in text else text
    matches = NUMBER_PATTERN.findall(candidate)
    if not matches:
        matches = NUMBER_PATTERN.findall(text)
    if not matches:
        return candidate.strip()
    return _canonicalize_number(matches[-1])


def _resolve_base_model_name(model_name_or_path: str | None, adapter_path: str | None) -> str:
    if model_name_or_path:
        return model_name_or_path
    if not adapter_path:
        raise ValueError("Either --model-name-or-path or --adapter-path must be provided.")

    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Could not infer base model without {adapter_config_path}")
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(f"adapter_config.json is missing base_model_name_or_path: {adapter_config_path}")
    return base_model_name


def _get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def load_model_and_tokenizer(
    model_name_or_path: str,
    adapter_path: str | None,
    torch_dtype_name: str,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    dtype = _torch_dtype_from_name(torch_dtype_name)
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "cuda"

    base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=True,
            autocast_adapter_dtype=False,
        )
    else:
        model = base_model
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    model.eval()
    if hasattr(model, "generation_config") and tokenizer.eos_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def get_parameter_summary(model: torch.nn.Module) -> dict[str, float | int]:
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable_params, all_params = model.get_nb_trainable_parameters()
    else:
        trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        all_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_percent = 100.0 * trainable_params / all_params if all_params else 0.0
    return {
        "trainable_params": int(trainable_params),
        "all_params": int(all_params),
        "trainable_percent": trainable_percent,
    }


def score_choice(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    continuation: str,
    max_length: int,
) -> float:
    model_device = _get_model_device(model)
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].to(model_device)
    full_ids = tokenizer(
        prompt + continuation,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )["input_ids"].to(model_device)
    continuation_ids = full_ids[:, prompt_ids.shape[1] :]
    if continuation_ids.numel() == 0:
        return float("-inf")

    with torch.no_grad():
        outputs = model(full_ids)
        log_probs = outputs.logits[:, :-1, :].log_softmax(dim=-1)

    start = prompt_ids.shape[1] - 1
    end = full_ids.shape[1] - 1
    token_log_probs = log_probs[:, start:end, :]
    gathered = token_log_probs.gather(-1, continuation_ids.unsqueeze(-1)).squeeze(-1)
    return gathered.sum().item()


def _apply_limit(dataset, limit: int | None):
    if limit is None or limit <= 0:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def evaluate_multiple_choice(
    name: str,
    dataset,
    prompt_builder,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    max_length: int,
    progress_every: int,
) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    correct = 0
    total = 0
    for index, row in enumerate(dataset):
        prompt, answer_labels, correct_index = prompt_builder(row)
        scores = [score_choice(model, tokenizer, prompt, f" {label}", max_length) for label in answer_labels]
        prediction = max(range(len(scores)), key=lambda item: scores[item])
        correct += int(prediction == correct_index)
        total += 1
        if progress_every and (index + 1) % progress_every == 0:
            print(f"[{name}] {index + 1}/{total} accuracy={correct / total:.4f}", flush=True)

    elapsed = time.perf_counter() - started
    peak_memory_gb = 0.0
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_reserved() / (1024**3)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "elapsed_seconds": elapsed,
        "max_memory_reserved_gb": peak_memory_gb,
    }


def build_mmlu_prompt(row) -> tuple[str, list[str], int]:
    labels = [chr(65 + index) for index in range(len(row["choices"]))]
    choices = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, row["choices"]))
    return f"{row['question']}\n{choices}\nAnswer:", labels, int(row["answer"])


def build_arc_prompt(row) -> tuple[str, list[str], int]:
    labels = [str(label) for label in row["choices"]["label"]]
    texts = [str(text) for text in row["choices"]["text"]]
    choices = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    correct_index = labels.index(str(row["answerKey"]))
    return f"{row['question']}\n{choices}\nAnswer:", labels, correct_index


def evaluate_mmlu(model, tokenizer, max_length: int, limit: int | None) -> dict[str, Any]:
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dataset = _apply_limit(dataset, limit)
    result = evaluate_multiple_choice("mmlu", dataset, build_mmlu_prompt, model, tokenizer, max_length, progress_every=200)
    result["split"] = "test"
    return result


def evaluate_arc_challenge(model, tokenizer, max_length: int, limit: int | None) -> dict[str, Any]:
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
    dataset = _apply_limit(dataset, limit)
    result = evaluate_multiple_choice(
        "arc_challenge",
        dataset,
        build_arc_prompt,
        model,
        tokenizer,
        max_length,
        progress_every=100,
    )
    result["split"] = "validation"
    return result


def evaluate_gsm8k(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    max_length: int,
    max_new_tokens: int,
    limit: int | None,
) -> dict[str, Any]:
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = _apply_limit(dataset, limit)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    correct = 0
    total = 0
    model_device = _get_model_device(model)
    for index, row in enumerate(dataset):
        prompt = f"Question: {row['question']}\nAnswer: Let's think step by step."
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model_device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        predicted_answer = extract_final_numeric_answer(generated_text)
        gold_answer = extract_final_numeric_answer(row["answer"])
        correct += int(predicted_answer == gold_answer)
        total += 1
        if (index + 1) % 100 == 0:
            print(f"[gsm8k] {index + 1}/{total} exact_match={correct / total:.4f}", flush=True)

    elapsed = time.perf_counter() - started
    peak_memory_gb = 0.0
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_reserved() / (1024**3)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "elapsed_seconds": elapsed,
        "max_memory_reserved_gb": peak_memory_gb,
        "split": "test",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--adapter-path")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["mmlu"],
        choices=BENCHMARK_CHOICES,
        help="Benchmarks to run. Default is MMLU-only for the first gate.",
    )
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-mmlu", type=int)
    parser.add_argument("--limit-arc-challenge", type=int)
    parser.add_argument("--limit-gsm8k", type=int)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_model_name = _resolve_base_model_name(args.model_name_or_path, args.adapter_path)
    model, tokenizer = load_model_and_tokenizer(base_model_name, args.adapter_path, args.torch_dtype)
    parameter_summary = get_parameter_summary(model)

    print(f"Base model: {base_model_name}")
    if args.adapter_path:
        print(f"Adapter: {args.adapter_path}")
    print(
        "Trainable params: "
        f"{parameter_summary['trainable_params']:,d} / {parameter_summary['all_params']:,d} "
        f"({parameter_summary['trainable_percent']:.4f}%)"
    )

    limit_mmlu = args.limit_mmlu if args.limit_mmlu is not None else args.limit
    limit_arc = args.limit_arc_challenge if args.limit_arc_challenge is not None else args.limit
    limit_gsm8k = args.limit_gsm8k if args.limit_gsm8k is not None else args.limit

    results: dict[str, Any] = {
        "model_name_or_path": base_model_name,
        "adapter_path": args.adapter_path,
        "parameters": parameter_summary,
        "benchmarks": {},
    }

    if "mmlu" in args.benchmarks:
        results["benchmarks"]["mmlu"] = evaluate_mmlu(model, tokenizer, args.max_length, limit_mmlu)
        print(
            f"MMLU accuracy: {results['benchmarks']['mmlu']['accuracy']:.4f} "
            f"({results['benchmarks']['mmlu']['correct']}/{results['benchmarks']['mmlu']['total']})"
        )
    if "arc_challenge" in args.benchmarks:
        results["benchmarks"]["arc_challenge"] = evaluate_arc_challenge(model, tokenizer, args.max_length, limit_arc)
        print(
            "ARC-Challenge accuracy: "
            f"{results['benchmarks']['arc_challenge']['accuracy']:.4f} "
            f"({results['benchmarks']['arc_challenge']['correct']}/{results['benchmarks']['arc_challenge']['total']})"
        )
    if "gsm8k" in args.benchmarks:
        results["benchmarks"]["gsm8k"] = evaluate_gsm8k(
            model,
            tokenizer,
            args.max_length,
            args.max_new_tokens,
            limit_gsm8k,
        )
        print(
            f"GSM8K exact match: {results['benchmarks']['gsm8k']['accuracy']:.4f} "
            f"({results['benchmarks']['gsm8k']['correct']}/{results['benchmarks']['gsm8k']['total']})"
        )

    accuracies = [entry["accuracy"] for entry in results["benchmarks"].values()]
    if accuracies:
        results["average_accuracy"] = sum(accuracies) / len(accuracies)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
