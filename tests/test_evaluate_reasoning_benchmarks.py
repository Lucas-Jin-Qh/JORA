from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script_module(name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


eval_script = _load_script_module("evaluate_reasoning_benchmarks", "scripts/evaluate_reasoning_benchmarks.py")


def test_extract_final_numeric_answer_prefers_hash_marker():
    assert eval_script.extract_final_numeric_answer("reasoning #### 1,234") == "1234"


def test_extract_final_numeric_answer_uses_last_number_without_hash_marker():
    prediction = "We try 12 first, but the correct answer is 42.0"
    assert eval_script.extract_final_numeric_answer(prediction) == "42"


def test_arc_prompt_uses_dataset_labels():
    row = {
        "question": "What is 2 + 2?",
        "choices": {"label": ["A", "B", "C", "D"], "text": ["1", "2", "4", "8"]},
        "answerKey": "C",
    }
    prompt, labels, correct_index = eval_script.build_arc_prompt(row)
    assert "A. 1" in prompt
    assert "C. 4" in prompt
    assert labels == ["A", "B", "C", "D"]
    assert correct_index == 2
