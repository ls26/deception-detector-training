"""Tests for the MMLU sycophancy multiple-choice task (tasks/multiple_choice.py).

The parity tests check the ported answer extraction and cue rendering against the
domain baseline's own artifacts under sycophancy_experiments/260819_domain_baseline;
they are skipped when that data is not present (e.g. CI outside the workspace).
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from obfuscation_atlas.config import TaskArgsConfig
from obfuscation_atlas.tasks.multiple_choice import (
    CUE_DIST,
    CUE_GOLD,
    CUE_TEMPLATES,
    compute_multiple_choice_rewards,
    cue_letter_for_condition,
    draw_cue,
    extract_answer,
    render_cued_user_content,
)

BASELINE_DIR = Path("/workspace/sycophancy_experiments/260819_domain_baseline")
needs_baseline_data = pytest.mark.skipif(
    not (BASELINE_DIR / "domain_questions.jsonl").exists(),
    reason="260819_domain_baseline data not available",
)


def _read_jsonl(path: Path, limit: int | None = None):
    with path.open() as f:
        lines = (json.loads(line) for line in f if line.strip())
        return list(itertools.islice(lines, limit))


def test_extract_answer_stages() -> None:
    assert extract_answer("Let me think. the answer is (A)") == ("A", "answer_is")
    assert extract_answer("the answer is B") == ("B", "answer_is")
    assert extract_answer("Answer: A") == ("A", "answer_colon")
    assert extract_answer("Either (A) or (B). I'll pick (A)") == ("A", "marked_letter")
    assert extract_answer("no letter here") == (None, "none")


def test_cue_letter_for_condition() -> None:
    assert cue_letter_for_condition(CUE_GOLD, "A") == "A"
    assert cue_letter_for_condition(CUE_DIST, "A") == "B"
    assert cue_letter_for_condition(CUE_DIST, "B") == "A"
    with pytest.raises(ValueError):
        cue_letter_for_condition("none", "A")


def test_task_args_validates_cue_gold_prob() -> None:
    TaskArgsConfig(cue_gold_prob=0.85)
    with pytest.raises(ValueError, match="cue_gold_prob"):
        TaskArgsConfig(cue_gold_prob=1.5)


def test_compute_multiple_choice_rewards() -> None:
    responses = [
        "reasoning... the answer is (A)",  # correct, followed cue
        "reasoning... the answer is (B)",  # wrong, did not follow cue
        "no parseable letter here",  # unparsed
        "the answer is (B)",  # correct, cue pointed at distractor A
    ]
    golds = ["A", "A", "A", "B"]
    cue_letters = ["A", "A", "A", "A"]
    conditions = [CUE_GOLD, CUE_GOLD, CUE_GOLD, CUE_DIST]

    scores, metrics = compute_multiple_choice_rewards(responses, golds, cue_letters, conditions)

    assert scores == [1.0, 0.0, 0.0, 1.0]
    assert metrics["rewards/task_success"] == [True, False, False, True]
    assert metrics["rewards_metrics/parse_failed"] == [False, False, True, False]
    assert metrics["rewards_metrics/followed_cue"] == [True, False, False, False]
    assert metrics["rewards_metrics/cue_gold"] == [True, True, True, False]


def test_compute_multiple_choice_rewards_without_cue_info() -> None:
    scores, metrics = compute_multiple_choice_rewards(["the answer is (A)"], ["A"])
    assert scores == [1.0]
    assert "rewards/task_success" in metrics
    assert "rewards_metrics/followed_cue" not in metrics
    assert "rewards_metrics/cue_gold" not in metrics


def test_draw_cue_deterministic_and_in_range() -> None:
    draws_a = [draw_cue(np.random.default_rng(42 + step + 2_000_000), 0.85) for step in range(20)]
    draws_b = [draw_cue(np.random.default_rng(42 + step + 2_000_000), 0.85) for step in range(20)]
    assert draws_a == draws_b  # same seed -> same served cues
    for condition, template_idx in draws_a:
        assert condition in (CUE_GOLD, CUE_DIST)
        assert 0 <= template_idx < len(CUE_TEMPLATES)
    # Degenerate probabilities pin the condition
    assert draw_cue(np.random.default_rng(0), 1.0)[0] == CUE_GOLD
    assert draw_cue(np.random.default_rng(0), 0.0)[0] == CUE_DIST


@needs_baseline_data
def test_extract_answer_parity_with_baseline_scores() -> None:
    """The ported extractor reproduces the baseline's stored predictions."""
    completions = _read_jsonl(BASELINE_DIR / "completions.jsonl", limit=3000)
    scores_rows = _read_jsonl(BASELINE_DIR / "scores.jsonl")
    # scores.jsonl rows are per (question, condition, cue_variant) with lists over samples
    preds = {}
    for row in scores_rows:
        for sample_idx, (pred, stage) in enumerate(zip(row["preds"], row["stages"])):
            preds[(row["question_id"], row["condition"], row["cue_variant"], sample_idx)] = (pred, stage)

    checked = 0
    for row in completions:
        key = (row["question_id"], row["condition"], row["cue_variant"], row["sample_idx"])
        if key not in preds:
            continue
        expected_pred, expected_stage = preds[key]
        pred, stage = extract_answer(row["completion"])
        assert pred == expected_pred, f"pred mismatch for {key}: {pred} != {expected_pred}"
        assert stage == expected_stage, f"stage mismatch for {key}: {stage} != {expected_stage}"
        checked += 1
    assert checked >= 1000, f"only {checked} rows checked; join with scores.jsonl failed?"


@needs_baseline_data
def test_cue_rendering_parity_with_baked_prompts() -> None:
    """render_cued_user_content reproduces the stored user_content_cue_* byte-for-byte."""
    questions = _read_jsonl(BASELINE_DIR / "domain_questions.jsonl")
    assert len(questions) == 2290
    for q in questions:
        gold = q["answer"]
        for condition in (CUE_GOLD, CUE_DIST):
            cue_letter = cue_letter_for_condition(condition, gold)
            rendered = render_cued_user_content(q["user_content_none"], cue_letter, q["cue_template_idx"])
            assert rendered == q[f"user_content_{condition}"], f"q{q['question_id']} {condition} mismatch"
        # The baked assigned cue letter matches the condition-derived one
        assert q["assigned_cue_letter"] == cue_letter_for_condition(q["assigned_condition"], gold)


@needs_baseline_data
def test_load_mmlu_sycophancy_dataset() -> None:
    from obfuscation_atlas.detectors.custom_dataset import load_mmlu_sycophancy_dataset

    dataset = load_mmlu_sycophancy_dataset()
    assert set(dataset.keys()) == {"train", "eval"}
    assert len(dataset["train"]) == 1603
    assert len(dataset["eval"]) == 687
    row = dataset["eval"][0]
    assert row["prompt"] == row["user_content_none"].rstrip() + "\n\n" + CUE_TEMPLATES[
        row["cue_template_idx"]
    ].format(letter=row["assigned_cue_letter"])
    assert row["answer"] in ("A", "B")
    assert len(row["completion"]) > 0
