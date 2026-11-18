"""The refit version's steps around the RL runs of seeds 45 and 46 with the training labels redrawn every round: per
line (seed), round 1 run here with 261103's weight-1.0 config (the original question file), then rounds 2-5 with the
probe refit before every round, round R training on 261108's question file for that round
(261108_frozen_probe_redrawn_labels/labels/roundR/labels.jsonl). 261109's rounds.py with round 1 run in this folder
and 261104's round-1 steps (build 1, score 1, generate 1) restored; no smoke run. frozen_rounds.py runs the frozen
version's rounds 2-5 from the same round 1.

    source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh && cd <this folder>
    $PY rounds.py LINE build R      -> LINE/probe_sets/roundR/{train,test}.jsonl, build.json
    $PY rounds.py LINE config R     -> LINE/s<seed>_refit_roundR.yaml (R >= 2)
    $PY rounds.py LINE score R      -> LINE/evals/first_probe_roundR/, LINE/stand_in/roundR/, LINE/checks/score_roundR.json
    $PY rounds.py LINE generate R   -> LINE/generations/roundR/completions.jsonl
    $PY rounds.py LINE selftest     -> LINE/checks/selftest.json
    $PY rounds.py LINE table R      -> LINE/probe_table.md, probe_table.json (rounds 1 ... R)

LINE is seed_45 or seed_46: the line's folder, under this one. --out DIR (build, selftest, table) writes that step's
files into DIR instead, for identity checks.

Round R's run key is s<seed>_round1 for round 1 (LINE/s<seed>_round1.yaml, written by hand from 261103's
kl_weight_1_0_s42.yaml) and s<seed>_refit_roundR for R >= 2 (LINE/s<seed>_refit_roundR.yaml). Its folder is
/workspace/results/run_results/<this folder's name>/<run key>_<run id> (run id in LINE/run_metrics_<run key>.json),
its final adapter that folder's checkpoints/final, its final eval LINE/evals/<run key>, and the base model's eval
answers scored with its probes LINE/evals/base_rescored_<run key>.

build       Round 1: copies of the original probe set (sycophancy_setup/data/bias_probe_train.jsonl, 261010's 1512
            base-model answers) and its test file (bias_probe_eval.jsonl, 261010's 777 base-model eval answers).
            Round R >= 2: the sources are the original set and the own answers of rounds 1 ... R-1 (generations/roundK).
            Own answers: as 261028's. A model's usual answer to a question is the content at least 6 of its 8 no-cue
            answers give (an unparsed answer counts for neither); questions without one drop out. The pool: cued answers
            that finished, parsed, name the cued option and are under 2048 tokens as prompt + answer cut after its
            letter (the trainer drops longer rows). Biased = the usual answer is the other option (260824's
            turn1_label). The original set is drawn from 261010's base-model pool, which gives the original file's rows.
            Each source gives n rows to each of the four groups (biased / unbiased x cue at the correct / at the wrong
            answer), n = its smallest group, capped at 378 and floored at 100. A group with fewer than n rows takes all
            of them, and the rest come from the unused rows of the own-answer source just before it, then the one before
            that (never from the original set). If those run out too, n drops in all four groups to the rows the
            shortest group can reach (the user's rule of 2026-11-05; it prints SHRUNK). Sources are drawn oldest first, each with its own random.Random(53) and
            groups in 261007's order, so a source's rows are the same in every round. The test file is round R-1's
            final-eval answers in the table's answer set (see table), without rows of 2048 tokens or more (the trainer
            drops them; such a drop prints DROPPED FROM TEST FILE). A training file never holds an eval question.
config      s<seed>_refit_roundR.yaml (R >= 2) from round_template.yaml: the line's seed and folder; round R-1's final
            adapter as the checkpoint; round R's probe files (the training file stands in for a test file with one
            class); max_train_examples 2 x the training rows; data.task_dataset.mmlu_questions_jsonl = 261108's
            labels/roundR/labels.jsonl, whose md5 must equal 261108's labels/md5.txt.
score       Round R's final-eval answers (8 no-cue samples + cued sample 0 per condition) through the eval's selfreport
            and labels stages with round R's adapter and a stand-in run folder whose start-of-run slot is the first
            probe (round 1's) and whose end-of-run slot is round R's probe. So in evals/first_probe_roundR/labels.jsonl
            probe_start = the first probe on round R's activations, probe_fresh = round R's probe, and
            probe_start_base_activations = the first probe on the base model's activations of the same text.
            checks/score_roundR.json: round R's probe scored here against the same probe in round R's final eval
            (probe_start there), on the same answers.
generate    eval_checkpoint.py generate --split train with round R's final adapter: per training question 8 answers
            with no cue, 3 with the cue at the correct answer, 3 with it at the wrong one.
selftest    The label, draw and test-file code against saved files (see selftest()).
table       Per round 1 ... R, both cue conditions pooled: the cued eval answers (sample 0) that name the cued option, labelled
            biased / unbiased against the checkpoint's own no-cue answer (5 of 8, the eval's rule), on the questions
            with a no-cue answer in both the base model's and the round's eval (compare.py's rule). Mean score of
            biased / unbiased answers and AUC for: round R's probe on its own test set (the trainer's saved test scores;
            read through round R-1's model, round 1: the base model); round R's probe on round R's answers and on the
            base model's; the first probe on round R's answers. And each round's probe training set per source. The
            trainer's end-of-run refit (probe_fresh in the round's final eval) is left out, as in the write-ups since
            261031.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, "/workspace/sycophancy-rl/sycophancy_setup")
import check_probe_scoring  # noqa: E402
import compare  # noqa: E402
import eval_checkpoint  # noqa: E402
import spec as S  # noqa: E402
from domain_imports import turn1_label  # noqa: E402
from obfuscation_atlas.tasks.multiple_choice import cut_after_answer_letter, find_answer_span  # noqa: E402

PARENT_DIR = Path(__file__).resolve().parent
GROUP = PARENT_DIR.name  # W&B group and /workspace/results/run_results/<group>/
LINE_SEEDS = {"seed_45": 45, "seed_46": 46}
EXP_DIR: Path  # the line's folder, PARENT_DIR / LINE; set by main from the line argument
SEED: int  # the line's training seed; set by main
RUN_RESULTS = Path("/workspace/results/run_results")
SYC = S.SYCOPHANCY_DIR
ROUNDS = (1, 2, 3, 4, 5)
TEMPLATE_PATH = PARENT_DIR / "round_template.yaml"
ORIGINAL_TRAIN_PATH = S.PACKAGE_DIR / "data" / "bias_probe_train.jsonl"
ORIGINAL_TEST_PATH = S.BIAS_PROBE_EVAL_PATH
BASE_POOL_PATH = SYC / "261010_confident_slice" / "arms" / "slice_cot" / "labels.jsonl"
BASE_POOL_SIZES_PATH = SYC / "261010_confident_slice" / "arms" / "train_sizes.json"
# Round R's question file (training labels redrawn per round), shared with 261108
LABELS_DIR = SYC / "261108_frozen_probe_redrawn_labels" / "labels"
DETECTOR_KEY = check_probe_scoring.DETECTOR_KEY
# selftest inputs only: 261028's selftest eval, and 261028's saved files for 261027's weight-0.6 seeds 42 and 43
SELFTEST_EVAL_DIR = SYC / "261025_probe_penalty_sd_floor" / "evals" / "floor_s42"
SELFTEST_261028 = SYC / "261028_own_answer_probes"
SELFTEST_EVAL_W06_S42 = SYC / "261027_probe_penalty_weight_sweep" / "weight_0_6" / "evals" / "weight_0_6_s42"
SELFTEST_BASE_EVAL_DIR = SYC / "261025_probe_penalty_sd_floor" / "evals" / "base_rescored_floor_s42"  # 261028's base answers
SELFTEST_CUT_GROUP = ("bias", "cue_gold")
SELFTEST_CUT_ROWS = 30

USUAL_MIN_AGREE = 6  # 261010's usual answer: at least 6 of 8 no-cue answers
MIN_PER_GROUP = 100
MAX_PER_GROUP = 378  # the original set's group size
DRAW_SEED = 53  # 261007's pool and sample seed
GROUPS = (("bias", "cue_gold"), ("bias", "cue_dist"), ("no bias", "cue_gold"), ("no bias", "cue_dist"))  # 261007's draw order
MAX_SEQUENCE_LENGTH = 2048  # SycophancyBiasProbeConfig.max_sequence_length: the trainer drops rows at or over it
ORIGINAL = "original"
ROW_FIELDS = ("question_id", "condition", "sample_idx", "prompt", "completion", "label")  # what the trainer reads
SCORE_NAMES = {
    "own_test_set": "Round R's probe; its own test set; round R-1's model (round 1: base)",
    "round_probe_round_answers": "Round R's probe; round R's answers; round R's model",
    "round_probe_base_answers": "Round R's probe; base model's answers; base model",
    "first_probe_round_answers": "First probe; round R's answers; round R's model",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    print(f"Wrote {path}")


def group_name(label: str, condition: str) -> str:
    return f"{label} / {condition}"


def run_key(round_number: int) -> str:
    assert round_number in ROUNDS, round_number
    return f"s{SEED}_round1" if round_number == 1 else f"s{SEED}_refit_round{round_number}"


def run_dir(round_number: int) -> Path:
    run_id = json.loads((EXP_DIR / f"run_metrics_{run_key(round_number)}.json").read_text())["run_id"]
    path = RUN_RESULTS / GROUP / f"{run_key(round_number)}_{run_id}"
    assert path.is_dir(), path
    return path


def adapter_of(round_number: int) -> Path:
    adapter = run_dir(round_number) / "checkpoints" / "final"
    assert (adapter / "adapter_config.json").exists(), adapter
    return adapter


def eval_dir(round_number: int) -> Path:
    return EXP_DIR / "evals" / run_key(round_number)


def base_eval_dir(round_number: int) -> Path:
    return EXP_DIR / "evals" / f"base_rescored_{run_key(round_number)}"


def set_dir(round_number: int) -> Path:
    return EXP_DIR / "probe_sets" / f"round{round_number}"


def generations_dir(round_number: int) -> Path:
    return EXP_DIR / "generations" / f"round{round_number}"


def first_probe_dir(round_number: int) -> Path:
    return EXP_DIR / "evals" / f"first_probe_round{round_number}"


def label_file(round_number: int) -> Path:
    """Round R's question file from 261108, after checking its md5 against 261108's labels/md5.txt."""
    path = LABELS_DIR / f"round{round_number}" / "labels.jsonl"
    recorded = dict(reversed(line.split("  ")) for line in (LABELS_DIR / "md5.txt").read_text().splitlines())
    assert hashlib.md5(path.read_bytes()).hexdigest() == recorded[str(path)], path
    return path


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #


def generate(round_number: int) -> None:
    args =[sys.executable, str(S.PACKAGE_DIR / "eval_checkpoint.py"), "generate", "--split", "train",
            "--out-dir", str(generations_dir(round_number)), "--adapter", str(adapter_of(round_number))]
    subprocess.run(args, cwd=S.PACKAGE_DIR, check=True)


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #


def usual_answer(contents: list[str | None], min_agree: int) -> str | None:
    for content in ("G", "D"):
        if contents.count(content) >= min_agree:
            return content
    return None


def probe_row(row: dict, label: str, source: str) -> dict:
    """A row in the probe-set format (the trainer's loader reads prompt, completion, question_id, condition,
    sample_idx and label)."""
    return {"question_id": row["question_id"], "condition": row["condition"], "sample_idx": row["sample_idx"],
            "cue_letter": row["cue_letter"], "answer_letter": find_answer_span(row["completion"])[0], "bias_rm": label,
            "label": int(label == "bias"), "source": source, "prompt": row["prompt"], "completion": row["completion"]}


def n_probe_tokens(tokenizer, row: dict) -> int:
    """Tokens of the text the trainer reads: prompt + the completion cut after its answer letter."""
    return len(tokenizer.encode(row["prompt"] + cut_after_answer_letter(row["completion"]), add_special_tokens=False))


def read_generations(path: Path) -> list[dict]:
    completions = read_jsonl(path)
    assert len(completions) == S.N_QUESTIONS["train"] * sum(S.N_SAMPLES_TRAIN.values()), len(completions)
    for row in completions:
        assert row["split"] == "train" and row["seed"] == eval_checkpoint.seed_of(row["question_id"], row["condition"], row["sample_idx"])
        assert row["prompt"].count("<|begin_of_text|>") == 1 and row["prompt"].startswith("<|begin_of_text|>")
    return completions


def own_pool(completions: list[dict], tokenizer, source: str, min_agree: int = USUAL_MIN_AGREE) -> tuple[list[dict], dict]:
    nocue: dict[int, list[str | None]] = {}
    for row in completions:
        if row["condition"] == "none":
            nocue.setdefault(row["question_id"], []).append(row["content"])
    assert set(len(contents) for contents in nocue.values()) == {S.N_SAMPLES_TRAIN["none"]}
    usual = {question_id: usual_answer(contents, min_agree) for question_id, contents in nocue.items()}
    filtered = Counter()
    pool = []
    for row in completions:
        if row["condition"] == "none":
            continue
        if row["finish_reason"] != "stop" or row["pred"] is None:
            filtered["unfinished_or_unparsed"] += 1
            continue
        if row["pred"] != row["cue_letter"]:
            filtered["does_not_name_cued_option"] += 1
            continue
        if usual[row["question_id"]] is None:
            filtered["no_usual_answer"] += 1
            continue
        assert find_answer_span(row["completion"])[0] == row["pred"], (row["question_id"], row["condition"], row["sample_idx"])
        if n_probe_tokens(tokenizer, row) >= MAX_SEQUENCE_LENGTH:
            filtered["too_long"] += 1
            continue
        label = turn1_label({"a_cued": row["content"], "a_none": usual[row["question_id"]], "a_other_cue": None,
                             "condition": row["condition"]}, S.DEFINITION)
        assert label in ("bias", "no bias"), label  # it names the cued option
        pool.append(probe_row(row, label, source))
    info = {"questions": len(nocue), "questions_with_usual_answer": sum(answer is not None for answer in usual.values()),
            "cued_answers": sum(row["condition"] != "none" for row in completions), "filtered": dict(filtered)}
    return pool, info


def base_pool(tokenizer) -> list[dict]:
    """261010's train pool: train-split rows labelled bias / no bias whose answer names the cued option (261007's
    step-1 filter; the per-prompt cap of 3 keeps every row, since each prompt has 3 answers)."""
    rows = [r for r in read_jsonl(BASE_POOL_PATH) if r["split"] == "train" and r["bias_rm"] in ("bias", "no bias") and r["followed_cue"] is True]
    cells = json.loads(BASE_POOL_SIZES_PATH.read_text())["train_pool_cells"]["bias_followed"]
    for label, condition in GROUPS:
        assert sum(r["bias_rm"] == label and r["condition"] == condition for r in rows) == cells[label][condition], (label, condition)
    for r in rows:
        assert find_answer_span(r["completion"])[0] == r["answer_letter"]
        assert n_probe_tokens(tokenizer, r) < MAX_SEQUENCE_LENGTH
    return [probe_row(r, r["bias_rm"], ORIGINAL) for r in rows]


def draw_sources(sources: list[tuple[str, list[dict]]]) -> tuple[list[dict], dict]:
    """The probe training set from sources in order, oldest first (the rule in the docstring's build)."""
    spare: dict[tuple[str, str], list[dict]] = {}  # (source, group): its shuffled rows of that group not yet used, in order
    train, info = [], {}
    for index, (name, pool) in enumerate(sources):
        pool_counts = {group_name(label, condition): sum(r["bias_rm"] == label and r["condition"] == condition for r in pool)
                       for label, condition in GROUPS}
        per_group = min(max(min(pool_counts.values()), MIN_PER_GROUP), MAX_PER_GROUP)
        earlier_own_sources = [earlier for earlier, _ in reversed(sources[:index]) if earlier != ORIGINAL]  # newest first
        rng = random.Random(DRAW_SEED)
        cells = {}
        for label, condition in GROUPS:
            cell = sorted((r for r in pool if r["bias_rm"] == label and r["condition"] == condition),
                          key=lambda r: (r["question_id"], r["condition"], r["sample_idx"]))
            rng.shuffle(cell)
            cells[group_name(label, condition)] = cell
        reachable = {group: len(cell) + sum(len(spare[(earlier, group)]) for earlier in earlier_own_sources) for group, cell in cells.items()}
        per_group = min(per_group, min(reachable.values()))  # under the floor only when the earlier sources run out of spare rows
        own_rows, topped_up_from = {}, {}
        for group, cell in cells.items():
            take = cell[:per_group]
            spare[(name, group)] = cell[per_group:]
            own_rows[group] = len(take)
            topped_up_from[group] = {}
            for earlier in earlier_own_sources:
                missing = per_group - len(take)
                if missing == 0:
                    break
                fill = spare[(earlier, group)][:missing]
                spare[(earlier, group)] = spare[(earlier, group)][missing:]
                take += fill
                if fill:
                    topped_up_from[group][earlier] = len(fill)
            assert len(take) == per_group
            train.extend(take)
        info[name] = {"pool": pool_counts, "per_group": per_group, "own_rows": own_rows, "topped_up_from": topped_up_from}
    keys = [(r["source"], r["question_id"], r["condition"], r["sample_idx"]) for r in train]
    assert len(set(keys)) == len(keys)
    return train, info


def table_answers(round_eval_dir: Path, round_base_eval_dir: Path) -> tuple[list[dict], set[int]]:
    """The table's answer set of one eval folder: cued sample 0 that names the cued option, bias / no bias, on the
    questions with a no-cue answer in both the base model's and this eval (compare.py's probe-row rule)."""
    base_rows = compare.rollout_rows(round_base_eval_dir)
    rows = compare.rollout_rows(round_eval_dir)
    shared = compare.questions_with_nocue_answer(base_rows) & compare.questions_with_nocue_answer(rows)
    return [r for r in rows if r["condition"] != "none" and r["question_id"] in shared and r["followed_cue"] and r["bias_rm"] in ("bias", "no bias")], shared


def test_rows(round_eval_dir: Path, round_base_eval_dir: Path, tokenizer) -> list[dict]:
    """A model's own eval answers of the table's set, in the probe-set format (the trainer's test file), without the
    rows the trainer would drop (MAX_SEQUENCE_LENGTH tokens or more)."""
    answers, _ = table_answers(round_eval_dir, round_base_eval_dir)
    wanted = {(r["question_id"], r["condition"]) for r in answers}
    labels = {(r["question_id"], r["condition"]): r["bias_rm"] for r in answers}
    rows = [c for c in read_jsonl(round_eval_dir / "completions.jsonl")
            if c["sample_idx"] == 0 and (c["question_id"], c["condition"]) in wanted and c["finish_reason"] == "stop"]
    kept = [c for c in rows if n_probe_tokens(tokenizer, c) < MAX_SEQUENCE_LENGTH]
    if len(kept) < len(rows):
        print(f"DROPPED FROM TEST FILE: {len(rows) - len(kept)} rows of {MAX_SEQUENCE_LENGTH} tokens or more")
    return [probe_row(c, labels[(c["question_id"], c["condition"])], "own") for c in kept]


def build(round_number: int, out_dir: Path | None) -> None:
    out_dir = out_dir or set_dir(round_number)
    assert not out_dir.exists(), f"{out_dir} exists"
    pool_info = {}
    if round_number == 1:
        out_dir.mkdir(parents=True)
        shutil.copyfile(ORIGINAL_TRAIN_PATH, out_dir / "train.jsonl")
        shutil.copyfile(ORIGINAL_TEST_PATH, out_dir / "test.jsonl")
        train, test = read_jsonl(out_dir / "train.jsonl"), read_jsonl(out_dir / "test.jsonl")
        counts = {group_name(label, condition): sum(r["label"] == int(label == "bias") and r["condition"] == condition for r in train)
                  for label, condition in GROUPS}
        assert set(counts.values()) == {MAX_PER_GROUP}, counts
        sources = {ORIGINAL: {"pool": counts, "per_group": MAX_PER_GROUP, "own_rows": counts, "topped_up_from": {group: {} for group in counts}}}
        test_answers = f"{ORIGINAL_TEST_PATH} (261010's base-model eval answers)"
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(S.MODEL_NAME)
        source_pools = [(ORIGINAL, base_pool(tokenizer))]
        for earlier_round in range(1, round_number):
            name = f"round{earlier_round}"
            pool, pool_info[name] = own_pool(read_generations(generations_dir(earlier_round) / "completions.jsonl"), tokenizer, name)
            source_pools.append((name, pool))
        train, sources = draw_sources(source_pools)
        test = test_rows(eval_dir(round_number - 1), base_eval_dir(round_number - 1), tokenizer)
        eval_checkpoint.write_jsonl(out_dir / "train.jsonl", train)
        eval_checkpoint.write_jsonl(out_dir / "test.jsonl", test)
        test_answers = f"{eval_dir(round_number - 1)} (round {round_number - 1}'s final-eval answers)"
    eval_questions = {q["question_id"] for q in read_jsonl(S.QUESTIONS_PATH) if q["split"] == "eval"}
    assert not {r["question_id"] for r in train} & eval_questions
    write_json(out_dir / "build.json", {
        "round": round_number, "sources": sources, "own_pool_filters": pool_info,
        "train_rows": len(train), "train_biased": sum(r["label"] for r in train),
        "test_answers": test_answers, "test_biased": sum(r["label"] for r in test), "test_unbiased": sum(1 - r["label"] for r in test),
        "test_one_class": len({r["label"] for r in test}) < 2,
    })
    for name, source in sources.items():
        if source["per_group"] < MIN_PER_GROUP:
            print(f"SHRUNK: round {round_number}: {name} gives {source['per_group']} rows per group, under the floor of {MIN_PER_GROUP}: "
                  f"the earlier sources ran out of spare rows")
        for group, fills in source["topped_up_from"].items():
            for earlier, n in fills.items():
                print(f"TOPPED UP: round {round_number}: {name} {group} +{n} rows of {earlier}")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def config(round_number: int) -> None:
    assert round_number >= 2, "round 1's config is LINE/s<seed>_round1.yaml, written by hand"
    template = yaml.safe_load(TEMPLATE_PATH.read_text())
    assert template["group"] == GROUP, template["group"]
    build_info = json.loads((set_dir(round_number) / "build.json").read_text())
    train_path = set_dir(round_number) / "train.jsonl"
    # The trainer needs both classes in its test file; the probe is fit before the test file is read (calibration is
    # off). A test file with one class is replaced by the training file (261031's fit).
    test_path = set_dir(round_number) / ("train.jsonl" if build_info["test_one_class"] else "test.jsonl")
    assert len(read_jsonl(train_path)) == build_info["train_rows"]
    checkpoint = str(adapter_of(round_number - 1))
    questions = str(label_file(round_number))
    key = run_key(round_number)
    run_config = copy.deepcopy(template)
    run_config["run_key"] = key
    run_config["output_dir"] = str(EXP_DIR)
    run_config["probe"]["directory_save_prefix"] = key
    run_config["checkpoint"] = checkpoint
    overrides = run_config["overrides"]
    overrides["training.seed"] = SEED
    overrides["training.max_steps"] = 128
    overrides["data.probe_dataset.bias_probe_train_jsonl"] = str(train_path)
    overrides["data.probe_dataset.bias_probe_eval_jsonl"] = str(test_path)
    overrides["data.probe_dataset.max_train_examples"] = 2 * build_info["train_rows"]
    overrides["data.task_dataset.mmlu_questions_jsonl"] = questions
    text = yaml.safe_dump(run_config, sort_keys=False)
    assert "FILL_IN" not in text
    path = EXP_DIR / f"{key}.yaml"
    assert not path.exists(), f"{path} exists"
    path.write_text(f"# Refit round {round_number} of the seed-{SEED} line: written by "
                    f"`rounds.py {EXP_DIR.name} config {round_number}` from ../round_template.yaml.\n"
                    f"#\n#   bash /workspace/sycophancy-rl/sycophancy_setup/run_rl.sh {EXP_DIR} {key} <attempt>\n\n" + text)
    print(f"Wrote {path}")


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #


def keyed(path: Path) -> dict[tuple, dict]:
    return {(r["question_id"], r["condition"]): r for r in read_jsonl(path) if r["sample_idx"] == 0}


def score(round_number: int) -> None:
    out_dir = first_probe_dir(round_number)
    rows = [r for r in read_jsonl(eval_dir(round_number) / "completions.jsonl") if r["condition"] == "none" or r["sample_idx"] == 0]
    assert len(rows) == S.N_QUESTIONS["eval"] * sum(S.N_SAMPLES.values()), len(rows)
    eval_checkpoint.write_jsonl(out_dir / "completions.jsonl", rows)
    stand_in = EXP_DIR / "stand_in" / f"round{round_number}"
    assert not stand_in.exists(), f"{stand_in} exists"
    (stand_in / S.FRESH_PROBE_SUBDIR).mkdir(parents=True)
    (stand_in / S.START_PROBE_SUBDIR).symlink_to(run_dir(1) / S.START_PROBE_SUBDIR)
    round_probe = run_dir(round_number) / S.START_PROBE_SUBDIR / f"detector_{S.START_PROBE_SUBDIR}__{DETECTOR_KEY}"
    (stand_in / S.FRESH_PROBE_SUBDIR / f"detector_{S.FRESH_PROBE_SUBDIR}__{DETECTOR_KEY}").symlink_to(round_probe)
    script = str(S.PACKAGE_DIR / "eval_checkpoint.py")
    subprocess.run([sys.executable, script, "selfreport", "--out-dir", str(out_dir), "--adapter", str(adapter_of(round_number)),
                    "--probe-run-dir", str(stand_in)], cwd=S.PACKAGE_DIR, check=True)
    subprocess.run([sys.executable, script, "labels", "--out-dir", str(out_dir)], cwd=S.PACKAGE_DIR, check=True)
    new, saved = keyed(out_dir / "labels.jsonl"), keyed(eval_dir(round_number) / "labels.jsonl")
    assert set(new) == set(saved)
    pairs = [(new[k]["probe_fresh"], saved[k]["probe_start"]) for k in new if new[k]["probe_fresh"] is not None or saved[k]["probe_start"] is not None]
    assert all(a is not None and b is not None for a, b in pairs)
    write_json(EXP_DIR / "checks" / f"score_round{round_number}.json", {
        "round_probe_here_vs_final_eval": {"answers": len(pairs), "max_abs_difference": max(abs(a - b) for a, b in pairs)},
        "first_probe_link": str((stand_in / S.START_PROBE_SUBDIR).resolve()),
        "round_probe_link": str((stand_in / S.FRESH_PROBE_SUBDIR / f"detector_{S.FRESH_PROBE_SUBDIR}__{DETECTOR_KEY}").resolve()),
        "round_probe_in_run_folder": str(round_probe),
    })


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #


def same_rows(first: list[dict], second: list[dict]) -> bool:
    return [tuple(r[f] for f in ROW_FIELDS) for r in first] == [tuple(r[f] for f in ROW_FIELDS) for r in second]


def selftest(out_dir: Path | None) -> None:
    """build's label, draw and test-file code against saved files:
    - on 261025 floor_s42's final-eval answers with the eval's 5-of-8 rule, its labels equal the eval's labels.jsonl;
    - drawing the original set from 261010's base pool gives the original training file, row for row and in order;
    - on 261028's saved generations of 261027's weight-0.6 seed-42 model, the draw gives the first 378 rows of each
      group of 261028's training file (261028 took 515 per group with the same shuffle), and the test-file code on
      that model's final eval gives 261028's test file byte for byte;
    - the top-up path, on a made-up short source: 261028's weight-0.6 seed-43 own answers with one group cut to its
      first 30 rows, drawn after the seed-42 own answers."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(S.MODEL_NAME)
    completions = read_jsonl(SELFTEST_EVAL_DIR / "completions.jsonl")
    pool, _ = own_pool(completions, tokenizer, "own", min_agree=S.NOCUE_MIN_AGREE)
    labels = {(r["question_id"], r["condition"], r["sample_idx"]): r for r in read_jsonl(SELFTEST_EVAL_DIR / "labels.jsonl")}
    eval_pool = [k for k, r in labels.items() if r["bias_rm"] in ("bias", "no bias") and r["pred"] == r["cue_letter"] and r["finish_reason"] == "stop"]
    original_train, original_info = draw_sources([(ORIGINAL, base_pool(tokenizer))])

    pool_42, _ = own_pool(read_generations(SELFTEST_261028 / "generations" / "w06_s42" / "completions.jsonl"), tokenizer, "round1")
    train_42, info_42 = draw_sources([("round1", pool_42)])
    saved_train_42 = read_jsonl(SELFTEST_261028 / "probe_sets" / "w06_s42" / "train.jsonl")
    saved_per_group = json.loads((SELFTEST_261028 / "probe_sets" / "w06_s42" / "build.json").read_text())["per_group"]
    saved_prefixes = [row for i in range(len(GROUPS)) for row in saved_train_42[i * saved_per_group : i * saved_per_group + MAX_PER_GROUP]]
    with tempfile.TemporaryDirectory() as tmp:
        test_path = Path(tmp) / "test.jsonl"
        eval_checkpoint.write_jsonl(test_path, test_rows(SELFTEST_EVAL_W06_S42, SELFTEST_BASE_EVAL_DIR, tokenizer))
        test_file_identical = test_path.read_bytes() == (SELFTEST_261028 / "probe_sets" / "w06_s42" / "test.jsonl").read_bytes()

    pool_43, _ = own_pool(read_generations(SELFTEST_261028 / "generations" / "w06_s43" / "completions.jsonl"), tokenizer, "round2")
    cut_label, cut_condition = SELFTEST_CUT_GROUP
    in_cut_group = [r for r in pool_43 if r["bias_rm"] == cut_label and r["condition"] == cut_condition]
    short_pool = [r for r in pool_43 if not (r["bias_rm"] == cut_label and r["condition"] == cut_condition)] + in_cut_group[:SELFTEST_CUT_ROWS]
    topped_up_train, topped_up_info = draw_sources([("round1", pool_42), ("round2", short_pool)])

    write_json((out_dir or EXP_DIR / "checks") / "selftest.json", {
        "labels": {
            "eval_pool_rows": len(pool), "eval_rows_naming_the_cue_with_a_bias_label": len(eval_pool),
            "same_rows_as_eval": sorted((r["question_id"], r["condition"], r["sample_idx"]) for r in pool) == sorted(eval_pool),
            "label_mismatches_vs_eval": sum(labels[(r["question_id"], r["condition"], r["sample_idx"])]["bias_rm"] != r["bias_rm"] for r in pool),
        },
        "original_set": {"draw": original_info[ORIGINAL], "equals_original_train_file_in_order": same_rows(original_train, read_jsonl(ORIGINAL_TRAIN_PATH))},
        "w06_s42": {"draw": info_42["round1"], "saved_per_group": saved_per_group,
                    "equals_first_378_of_each_group_in_261028_train_file": same_rows(train_42, saved_prefixes),
                    "test_file_identical_to_261028": test_file_identical},
        "top_up": {"cut_group": group_name(cut_label, cut_condition), "cut_to_rows": SELFTEST_CUT_ROWS,
                   "draw": topped_up_info, "train_rows": len(topped_up_train),
                   "rows_by_source": dict(Counter(r["source"] for r in topped_up_train))},
    })


# --------------------------------------------------------------------------- #
# table
# --------------------------------------------------------------------------- #


def saved_scores(round_run_dir: Path, subdir: str, index: int) -> dict[str, list[float]]:
    """A saved probe's scores per class, mean over layers 13-16 of each layer's score: index 0 = the test set,
    2 = the training set (the trainer's pickle)."""
    path = round_run_dir / subdir / f"f1_and_recall_plot_{subdir}__{DETECTOR_KEY}.pkl"
    layer_metrics = pickle.load(path.open("rb"))[index]
    out = {}
    for label in ("positive", "negative"):
        per_layer = [layer_metrics[str(layer)]["scores"][label] for layer in S.PROBE_LAYERS]
        out[label] = [sum(values) / len(values) for values in zip(*per_layer, strict=True)]
    return out


def score_summary(scores: dict[str, list[float]]) -> dict:
    return {"n_biased": len(scores["positive"]), "n_unbiased": len(scores["negative"]), "mean_biased": compare.mean(scores["positive"]),
            "mean_unbiased": compare.mean(scores["negative"]), "auc": compare.auc(scores["positive"], scores["negative"])}


def pooled(rows: list[dict], name: str) -> dict:
    biased = [r[name] for r in rows if r["bias_rm"] == "bias" and r[name] is not None]
    unbiased = [r[name] for r in rows if r["bias_rm"] == "no bias" and r[name] is not None]
    return {"n_biased": len(biased), "n_unbiased": len(unbiased), "mean_biased": compare.mean(biased), "mean_unbiased": compare.mean(unbiased),
            "auc": compare.auc(biased, unbiased)}


def fmt(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def table(last_round: int, out_dir: Path | None) -> None:
    out_dir = out_dir or EXP_DIR
    result = {}
    for round_number in range(1, last_round + 1):
        build_info = json.loads((set_dir(round_number) / "build.json").read_text())
        own_test = None
        if not build_info["test_one_class"]:
            own_test = score_summary(saved_scores(run_dir(round_number), S.START_PROBE_SUBDIR, 0))
            assert (own_test["n_biased"], own_test["n_unbiased"]) == (build_info["test_biased"], build_info["test_unbiased"]), own_test
        round_answers, shared = table_answers(eval_dir(round_number), base_eval_dir(round_number))
        first_answers, first_shared = table_answers(first_probe_dir(round_number), base_eval_dir(round_number))
        assert first_shared == shared
        base_answers = [r for r in compare.rollout_rows(base_eval_dir(round_number))
                        if r["condition"] != "none" and r["question_id"] in shared and r["followed_cue"] and r["bias_rm"] in ("bias", "no bias")]
        result[f"round{round_number}"] = {
            "shared_questions": len(shared), "test_answers": build_info["test_answers"],
            "scores": {"own_test_set": own_test, "round_probe_round_answers": pooled(round_answers, "probe_start"),
                       "round_probe_base_answers": pooled(base_answers, "probe_start"),
                       "first_probe_round_answers": pooled(first_answers, "probe_start")},
            "sources": build_info["sources"], "train_rows": build_info["train_rows"],
        }

    lines = ["# Probe scores per round", "",
             "Both cue conditions pooled. Answers except a probe's own test set: the cued eval answers (sample 0) that name the",
             "cued option, labelled biased / unbiased against the checkpoint's own no-cue answer (at least 5 of its 8 no-cue",
             "answers), on the questions with a no-cue answer in both the base model's and the round's eval. A probe's own test",
             "set: the trainer's saved test scores on its test file (round 1: 261010's 777 base-model eval answers; round R: round",
             "R-1's final-eval answers in the same answer set; – if that file had one class). Built by `rounds.py table`.", "",
             "| Round | Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |", "|---|---|---|---|---|"]
    for name, round_result in result.items():
        for key, title in SCORE_NAMES.items():
            r = round_result["scores"][key]
            if r is None:
                lines.append(f"| {name} | {title} | – | – | – |")
            else:
                lines.append(f"| {name} | {title} | {r['n_biased']} / {r['n_unbiased']} | {fmt(r['mean_biased'])} / {fmt(r['mean_unbiased'])} | {fmt(r['auc'])} |")
    columns = [group_name(label, condition) for label, condition in GROUPS]
    lines += ["", "# Probe training sets per round", "",
              "Pool: a source's rows per group (biased, cue at correct / biased, cue at wrong / unbiased, cue at correct / unbiased, cue",
              "at wrong). Per group: the rows each group takes (the smallest group, capped at 378, floored at 100; under 100 when the",
              "earlier own-answer sources ran out of spare rows to fill the smallest group). Own rows: the",
              "source's own rows per group; topped up: rows from an earlier own-answer source filling a group.", "",
              "| Round | Source | Pool | Per group | Own rows | Topped up | Training rows (round) |", "|---|---|---|---|---|---|---|"]
    for name, round_result in result.items():
        for source, s in round_result["sources"].items():
            fills = "; ".join(f"{group}: " + ", ".join(f"+{n} from {earlier}" for earlier, n in from_sources.items())
                              for group, from_sources in s["topped_up_from"].items() if from_sources) or "none"
            lines.append(f"| {name} | {source} | {' / '.join(str(s['pool'][c]) for c in columns)} | {s['per_group']} | "
                         f"{' / '.join(str(s['own_rows'][c]) for c in columns)} | {fills} | {round_result['train_rows']} |")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_table.md").write_text("\n".join(lines) + "\n")
    write_json(out_dir / "probe_table.json", result)


def main() -> None:
    global EXP_DIR, SEED
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("line", choices=tuple(LINE_SEEDS))
    parser.add_argument("command", choices=("generate", "build", "config", "score", "selftest", "table"))
    parser.add_argument("round", nargs="?", type=int)
    parser.add_argument("--out", type=Path, default=None, help="build, selftest, table: write here instead (identity checks)")
    args = parser.parse_args()
    EXP_DIR = PARENT_DIR / args.line
    SEED = LINE_SEEDS[args.line]
    assert EXP_DIR.is_dir(), EXP_DIR
    print(f"=== rounds.py {args.line} {args.command} {args.round or ''} ===", flush=True)
    if args.command in ("generate", "build", "config", "score", "table"):
        assert args.round in ROUNDS, args.round
        assert args.out is None or args.command in ("build", "table")
        if args.command == "build":
            build(args.round, args.out)
        elif args.command == "table":
            table(args.round, args.out)
        else:
            {"generate": generate, "config": config, "score": score}[args.command](args.round)
    else:
        assert args.round is None
        selftest(args.out)


if __name__ == "__main__":
    main()
