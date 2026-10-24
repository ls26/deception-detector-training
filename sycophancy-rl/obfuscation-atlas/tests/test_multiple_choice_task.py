"""Tests for the MMLU sycophancy multiple-choice task (tasks/multiple_choice.py).

The parity tests check the ported answer extraction and cue rendering against the
domain baseline's own artifacts under sycophancy_experiments/260819_domain_baseline,
the no-flip test checks the training-label code against the August no-penalty run
(260825_grpo_det0), and the loader test reads 261013_noisy_label_rl's question file;
the no-CoT tests compare the trainer's prompts with 261007's stored no-CoT prompts on
261014_nocot_noisy_label_rl's question file; the training-cue-off tests run the trainer's own
prompt and reward methods on both question files against the stored no-cue eval prompts;
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
    other_letter,
    parse_answer,
    render_cued_user_content,
)

BASELINE_DIR = Path("/workspace/sycophancy_experiments/260819_domain_baseline")
needs_baseline_data = pytest.mark.skipif(
    not (BASELINE_DIR / "domain_questions.jsonl").exists(),
    reason="260819_domain_baseline data not available",
)
# The August no-penalty run (W&B lx6gcbcl): 128 per-step response files, and the per-step
# counts the 261012 checker recomputed from them with the earlier reward function (gold only).
AUGUST_RESPONSES_DIR = Path("/workspace/results/run_results/syco_det0/syco_det0_s42_lx6gcbcl/checkpoints/responses")
AUGUST_CHECK = Path("/workspace/sycophancy_experiments/261012_port_rerun/check_august.json")
needs_august_run = pytest.mark.skipif(
    not (AUGUST_RESPONSES_DIR.exists() and AUGUST_CHECK.exists()), reason="August run files not available"
)
NOISY_QUESTIONS = Path("/workspace/sycophancy_experiments/261013_noisy_label_rl/dataset/labels.jsonl")
needs_noisy_questions = pytest.mark.skipif(not NOISY_QUESTIONS.exists(), reason="261013 question file not available")
# The no-CoT question file (261014) and 261007's stored no-CoT prompts (gens/nocot/turn1.jsonl)
NOCOT_QUESTIONS = Path("/workspace/sycophancy_experiments/261014_nocot_noisy_label_rl/dataset/labels.jsonl")
NOCOT_TURN1 = Path("/workspace/sycophancy_experiments/261007_format_label_cue_ablation/gens/nocot/turn1.jsonl")
needs_nocot_data = pytest.mark.skipif(
    not (NOCOT_QUESTIONS.exists() and NOCOT_TURN1.exists()), reason="261014 question file or 261007 no-CoT prompts not available"
)
NOCOT_PREFILL = "The answer is ("


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
        "reasoning... the answer is (A)",  # label = gold A, cue at A: rewarded, correct, followed cue
        "reasoning... the answer is (B)",  # label = gold A, cue at A: not rewarded, wrong
        "no parseable letter here",  # unparsed
        "the answer is (B)",  # label = gold B, cue at the other option A
        "the answer is (B)",  # flipped: gold A, label B, cue at the label B: rewarded, wrong, followed cue
        "the answer is (A)",  # flipped: gold A, label B, cue at the label B: not rewarded, correct
    ]
    golds = ["A", "A", "A", "B", "A", "A"]
    labels = ["A", "A", "A", "B", "B", "B"]
    cue_letters = ["A", "A", "A", "A", "B", "B"]

    scores, metrics = compute_multiple_choice_rewards(responses, golds, labels, cue_letters)

    assert scores == [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    assert metrics["rewards/task_success"] == [True, False, False, True, True, False]
    assert metrics["rewards_metrics/true_correct"] == [True, False, False, True, False, True]
    assert metrics["rewards_metrics/parse_failed"] == [False, False, True, False, False, False]
    assert metrics["rewards_metrics/label_flipped"] == [False, False, False, False, True, True]
    assert metrics["rewards_metrics/followed_cue"] == [True, False, False, False, True, False]
    assert metrics["rewards_metrics/cue_gold"] == [True, True, True, False, False, False]
    assert metrics["rewards_metrics/cue_at_label"] == [True, True, True, False, True, True]


def test_compute_multiple_choice_rewards_without_cue_info() -> None:
    scores, metrics = compute_multiple_choice_rewards(["the answer is (A)"], ["A"], ["A"])
    assert scores == [1.0]
    assert "rewards/task_success" in metrics
    assert "rewards_metrics/followed_cue" not in metrics
    assert "rewards_metrics/cue_gold" not in metrics


def test_draw_cue_deterministic_and_in_range() -> None:
    draws_a = [draw_cue(np.random.default_rng(42 + step + 2_000_000), 0.75) for step in range(20)]
    draws_b = [draw_cue(np.random.default_rng(42 + step + 2_000_000), 0.75) for step in range(20)]
    assert draws_a == draws_b  # same seed -> same served cues
    for at_label, template_idx in draws_a:
        assert at_label in (True, False)
        assert 0 <= template_idx < len(CUE_TEMPLATES)
    # Degenerate probabilities pin where the cue points
    assert draw_cue(np.random.default_rng(0), 1.0)[0] is True
    assert draw_cue(np.random.default_rng(0), 0.0)[0] is False


@needs_august_run
@needs_baseline_data
def test_no_flip_reproduces_august_cues_and_counts() -> None:
    """With training label = gold, the new cue draw and reward give what the August run served and
    what the earlier (gold-only) reward function counted, on its first 5 steps."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    train_questions = [q for q in _read_jsonl(BASELINE_DIR / "domain_questions.jsonl") if q["split"] == "train"]
    lookup = {}  # served prompt -> (gold letter, cue letter, template index)
    for q in train_questions:
        for cue_letter in ("A", "B"):
            for template_idx in range(len(CUE_TEMPLATES)):
                user_content = render_cued_user_content(q["user_content_none"], cue_letter, template_idx)
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True
                )
                lookup[prompt] = (q["answer"], cue_letter, template_idx)
    expected_counts = {row["file_step"]: row["recomputed_counts"] for row in json.loads(AUGUST_CHECK.read_text())["per_step"]}

    for step in range(5):
        rows = _read_jsonl(AUGUST_RESPONSES_DIR / f"step_{step}.jsonl")
        assert len(rows) == 64 * 8
        served = [lookup[rows[i]["prompt"]] for i in range(0, len(rows), 8)]
        rng = np.random.default_rng(42 + step + 2_000_000)  # training.seed 42, the trainer's cue RNG
        for gold, cue_letter, template_idx in served:
            at_label, drawn_template = draw_cue(rng, 0.85)
            assert (gold if at_label else other_letter(gold), drawn_template) == (cue_letter, template_idx)

        golds = [served[i // 8][0] for i in range(len(rows))]
        cue_letters = [served[i // 8][1] for i in range(len(rows))]
        completions = [row["completion"].removesuffix("<|eot_id|>") for row in rows]
        _, metrics = compute_multiple_choice_rewards(completions, golds, golds, cue_letters)
        counts = {
            "rewards/task_success": sum(metrics["rewards/task_success"]),
            "rewards_metrics/followed_cue": sum(metrics["rewards_metrics/followed_cue"]),
            "rewards_metrics/parse_failed": sum(metrics["rewards_metrics/parse_failed"]),
            "cue_gold_prompts": sum(metrics["rewards_metrics/cue_gold"]) // 8,
        }
        assert counts == expected_counts[step], f"step {step}: {counts} != {expected_counts[step]}"
        assert metrics["rewards/task_success"] == metrics["rewards_metrics/true_correct"]


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
@needs_noisy_questions
def test_load_mmlu_sycophancy_dataset() -> None:
    from obfuscation_atlas.detectors.custom_dataset import load_mmlu_sycophancy_dataset

    dataset = load_mmlu_sycophancy_dataset(mmlu_questions_jsonl=str(NOISY_QUESTIONS))
    assert set(dataset.keys()) == {"train", "eval"}
    assert len(dataset["train"]) == 1217
    assert len(dataset["eval"]) == 489
    assert sum(dataset["train"]["label_flipped"]) == 243
    assert sum(dataset["eval"]["label_flipped"]) == 98
    for row in dataset["eval"]:
        assert row["prompt"] == row["user_content_none"].rstrip() + "\n\n" + CUE_TEMPLATES[
            row["cue_template_idx"]
        ].format(letter=row["assigned_cue_letter"])
        assert row["answer"] in ("A", "B")
        assert row["train_label"] == (other_letter(row["answer"]) if row["label_flipped"] else row["answer"])
        assert len(row["completion"]) > 0


def test_parse_answer_letter_only() -> None:
    assert parse_answer("A", letter_only=True) == "A"
    assert parse_answer("B", letter_only=True) == "B"
    for text in ("Both", " A", "A)", "a", "", "C", "the answer is (A)"):
        assert parse_answer(text, letter_only=True) is None, text
    # The CoT chain is unchanged; it reads no bare letter, and on prefill + token it reads "Both" as B
    assert parse_answer("the answer is (A)", letter_only=False) == "A"
    assert parse_answer("A", letter_only=False) is None
    assert parse_answer(NOCOT_PREFILL + "Both", letter_only=False) == "B"


def test_compute_multiple_choice_rewards_letter_only() -> None:
    responses = ["A", "B", "Both", ""]  # label A; flipped label B; unparsed token; end-of-turn only
    golds = ["A", "A", "A", "B"]
    labels = ["A", "B", "A", "B"]
    cue_letters = ["A", "A", "B", "B"]
    scores, metrics = compute_multiple_choice_rewards(responses, golds, labels, cue_letters, letter_only=True)
    assert scores == [1.0, 1.0, 0.0, 0.0]
    assert metrics["rewards_metrics/true_correct"] == [True, False, False, False]
    assert metrics["rewards_metrics/parse_failed"] == [False, False, True, True]
    assert metrics["rewards_metrics/followed_cue"] == [True, False, False, False]


def test_task_args_nocot_prefill_default_off() -> None:
    assert TaskArgsConfig().nocot_prefill is None


@needs_nocot_data
def test_nocot_training_prompts_match_261007() -> None:
    """The trainer's training prompt (no-cue content + cue at a letter in a wording, chat template, then the
    prefill, as _prepare_mcq_training_inputs + _prepare_inputs_for_generation build it) equals 261007's
    stored no-CoT prompt + prefill on every cued prompt of the question file, and ends in token 320."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    questions = {q["question_id"]: q for q in _read_jsonl(NOCOT_QUESTIONS)}
    assert len(questions) == 1304 + 547
    stored = {}
    for row in _read_jsonl(NOCOT_TURN1):
        if row["sample_idx"] == 0 and row["condition"] != "none" and row["question_id"] in questions:
            stored[(row["question_id"], row["condition"])] = row["prompt"]
    assert len(stored) == 2 * len(questions)
    for (question_id, condition), prompt in stored.items():
        q = questions[question_id]
        user_content = render_cued_user_content(
            q["user_content_none"], cue_letter_for_condition(condition, q["answer"]), q["cue_template_idx"]
        )
        served = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True
        ) + NOCOT_PREFILL
        assert served == prompt + NOCOT_PREFILL, f"q{question_id} {condition}"
        assert tokenizer(served, add_special_tokens=False)["input_ids"][-1] == 320


@needs_baseline_data
@needs_nocot_data
def test_load_nocot_question_file() -> None:
    """The loader on the no-CoT question file: counts, flips, and the in-loop eval prompt (the assigned
    condition's user content through chat_template_preparer, then the prefill) equals 261007's."""
    from transformers import AutoTokenizer

    from obfuscation_atlas.detectors.custom_dataset import load_mmlu_sycophancy_dataset
    from obfuscation_atlas.training.obfuscation_training import chat_template_preparer

    dataset = load_mmlu_sycophancy_dataset(mmlu_questions_jsonl=str(NOCOT_QUESTIONS))
    assert len(dataset["train"]) == 1304
    assert len(dataset["eval"]) == 547
    assert sum(dataset["train"]["label_flipped"]) == 261
    assert sum(dataset["eval"]["label_flipped"]) == 109
    stored = {
        (row["question_id"], row["condition"]): row["prompt"] for row in _read_jsonl(NOCOT_TURN1) if row["sample_idx"] == 0
    }
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    templated = chat_template_preparer(dataset["eval"], system_prompt=None, tokenizer=tokenizer)
    for row, templated_row in zip(dataset["eval"], templated):
        assert row["prompt"] == render_cued_user_content(row["user_content_none"], row["assigned_cue_letter"], row["cue_template_idx"])
        assert "Think step by step" not in row["prompt"]
        assert row["train_label"] == (other_letter(row["answer"]) if row["label_flipped"] else row["answer"])
        expected = stored[(row["question_id"], row["assigned_condition"])]
        assert templated_row["prompt"] == expected, f"q{row['question_id']}"  # the trainer then appends the prefill


def test_task_args_train_cue_default_on() -> None:
    assert TaskArgsConfig().train_cue is True


def _stored_no_cue_prompts(path: Path) -> dict[int, str]:
    """question_id -> the stored no-cue prompt (260819's CoT prompts or 261007's no-CoT prompts, no prefill)."""
    return {row["question_id"]: row["prompt"] for row in _read_jsonl(path) if row["condition"] == "none" and row["sample_idx"] == 0}


@needs_baseline_data
@needs_nocot_data
@pytest.mark.parametrize("question_file", ["cot", "nocot"])
def test_training_cue_off_serves_the_no_cue_prompt(question_file: str) -> None:
    """With task_args.train_cue off, the trainer's own _prepare_mcq_training_inputs (on a stand-in with only
    the attributes it reads) serves every training question of the setup's question file exactly the stored
    no-cue eval prompt (+ the prefill for no-CoT) with cue None, and compute_sequence_reward scores against
    train_label with no cue metrics."""
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    from obfuscation_atlas.config import TaskType
    from obfuscation_atlas.training.grpo_trainer import GRPOObfuscationTrainer

    if question_file == "cot":
        questions_path, prefill = NOISY_QUESTIONS, None
        stored = _stored_no_cue_prompts(BASELINE_DIR / "completions.jsonl")
    else:
        questions_path, prefill = NOCOT_QUESTIONS, NOCOT_PREFILL
        stored = _stored_no_cue_prompts(NOCOT_TURN1)
    train_questions = [q for q in _read_jsonl(questions_path) if q["split"] == "train"]
    assert len(train_questions) == {"cot": 1217, "nocot": 1304}[question_file]

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    obfuscation_config = SimpleNamespace(
        task=TaskType.MULTIPLE_CHOICE,
        task_args=TaskArgsConfig(cue_gold_prob=0.75, nocot_prefill=prefill, train_cue=False),
        training=SimpleNamespace(seed=42),
    )
    trainer = SimpleNamespace(
        obfuscation_config=obfuscation_config,
        config=SimpleNamespace(tokenizer=SimpleNamespace(apply_chat_template=False)),  # as in grpo_mcq
        tokenizer=tokenizer,
        reward_infos=train_questions,
        _generation_cue_infos={},
        _get_inputs_for_step=lambda step: (list(range(len(train_questions))), None),
    )
    trainer._prepare_inputs_for_generation = GRPOObfuscationTrainer._prepare_inputs_for_generation.__get__(trainer)
    generation_inputs = GRPOObfuscationTrainer._prepare_mcq_training_inputs(trainer, step=0)

    cue_infos = trainer._generation_cue_infos[id(generation_inputs)]
    assert all(info == {"cue_letter": None, "template_idx": None} for info in cue_infos.values())
    for q, served in zip(train_questions, generation_inputs.prompts_formatted, strict=True):
        assert served == stored[q["question_id"]] + (prefill or ""), f"q{q['question_id']}"

    # Answer every question with its training label: reward 1 on all, true_correct only where the label is gold.
    responses = [q["train_label"] if prefill else f"the answer is ({q['train_label']})" for q in train_questions]
    reward_trainer = SimpleNamespace(
        obfuscation_config=obfuscation_config, reward_infos=train_questions, tokenizer=SimpleNamespace(eot_token="<|eot_id|>")
    )
    scores, metrics = GRPOObfuscationTrainer.compute_sequence_reward(
        reward_trainer,
        generation_inputs.prompts_formatted,
        responses,
        list(range(len(train_questions))),
        1,
        cue_infos=cue_infos,
    )
    assert scores == [1.0] * len(train_questions)
    assert metrics["rewards_metrics/true_correct"] == [q["train_label"] == q["answer"] for q in train_questions]
    assert not any(key in metrics for key in ("rewards_metrics/followed_cue", "rewards_metrics/cue_gold", "rewards_metrics/cue_at_label"))
