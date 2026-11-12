"""The steps around the RL runs of 261107's frozen-probe rounds 2-5 rerun with the training labels redrawn every
round: two lines (seeds 42 and 43), each continuing 261103's weight-1.0 run of its seed as round 1, rounds 2-5 against
round 1's probe (never refit), round R training on its own question file labels/roundR/labels.jsonl. 261107's
rounds.py with a labels command and the question file set per round.

    source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh && cd <this folder>
    $PY rounds.py labels                 -> labels/round{2..5}/labels.jsonl, labels/md5.txt, labels/summary.json
    $PY rounds.py labels --known-answer  (writes nothing; asserts the builder reproduces the original question file)
    $PY rounds.py LINE config R          -> LINE/s<seed>_redrawn_roundR.yaml (R >= 2; round 2 also LINE/smoke_s<seed>_redrawn_round2.yaml)
    $PY rounds.py LINE links R           -> LINE/links/roundR/
    $PY rounds.py LINE penalty_check RUN_YAML PENALTY_PROBE_RUN OTHER_PROBE_RUN   -> LINE/checks/penalty_<run key>.json

LINE is seed_42 or seed_43: the line's folder, under this one.

Round 1 of a line is 261103's weight-1.0 run of its seed (ROUND1_RUNS), not rerun: its folder is
/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s<seed>_<run id>. Its start-of-run probe (fit on the base
model's answers through the base model) is the line's frozen probe. Round R >= 2's run is LINE/s<seed>_redrawn_roundR.yaml;
its folder is /workspace/results/run_results/<this folder's name>/s<seed>_redrawn_roundR_<run id> (run id in
LINE/run_metrics_s<seed>_redrawn_roundR.json), and its final adapter that folder's checkpoints/final.

The trainer still fits its own start-of-run probe (on round 1's probe files, through round R-1's model) and saves it in
the run folder's start-of-run slot, then swaps in the frozen probe for the penalty
(train_grpo_model_against_probe.py:553-575). So the run folder's start-of-run slot holds a probe the run never used.

labels          The question file of each round 2-5, shared by both lines: the original question file
                (spec.QUESTIONS_PATH, built by 261013_noisy_label_rl/build_dataset.py) with a new set of flipped
                training questions, exactly N_FLIPPED_TRAIN of the 1217, drawn as the original draw was
                (random.Random(seed).sample(sorted train question ids, 243)) with seed REDRAW_SEED + R. On train rows
                train_label, label_flipped, assigned_cue_letter and assigned_condition are set as build_dataset.py
                sets them; every other field, and every eval row, stays as in the original. With --known-answer, the
                same builder with the original seed (261013) must give the original file byte for byte.
config          s<seed>_redrawn_roundR.yaml (R >= 2) from round_template.yaml: the line's seed and folder; round R-1's
                final adapter as the checkpoint; detector.frozen_probe_weights_dir = round 1's start-of-run probe folder;
                data.task_dataset.mmlu_questions_jsonl = labels/roundR/labels.jsonl.
                smoke_s<seed>_redrawn_round2.yaml (round 2): the same with 2 steps and an in-loop eval after each.
links           LINE/links/roundR/, the folder the final eval reads in place of the run folder (eval_checkpoint.py
                selfreport --probe-run-dir): its start-of-run entry links to round 1's start-of-run probe (the penalty
                probe), its end-of-run entry to round R's own end-of-run refit. So in the eval's labels.jsonl
                probe_start = the frozen probe, probe_fresh = this run's end-of-run refit, as for any other run.
penalty_check   Every parsed answer of RUN_YAML's run's first training step (checkpoints/responses/step_0.jsonl, the
                answers the policy at step 0 gave, which is the config's checkpoint) re-scored outside the trainer:
                prompt + answer through the checkpoint, read at the answer letter's token, with the start-of-run probe
                of PENALTY_PROBE_RUN (the probe the penalty should have used) and of OTHER_PROBE_RUN (a probe it should
                not have used), each against the probe_score the trainer saved. Reports max / median / mean absolute
                differences and the number above 0.02; the review applies the rule.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, "/workspace/sycophancy-rl/sycophancy_setup")
import check_probe_scoring  # noqa: E402
import eval_checkpoint  # noqa: E402
import spec as S  # noqa: E402
from obfuscation_atlas.tasks.multiple_choice import find_answer_span  # noqa: E402

PARENT_DIR = Path(__file__).resolve().parent
GROUP = PARENT_DIR.name  # W&B group and /workspace/results/run_results/<group>/
LINE_SEEDS = {"seed_42": 42, "seed_43": 43}
EXP_DIR: Path  # the line's folder, PARENT_DIR / LINE; set by main from the line argument
SEED: int  # the line's training seed; set by main
RUN_RESULTS = Path("/workspace/results/run_results")
ROUNDS = (1, 2, 3, 4, 5)
TEMPLATE_PATH = PARENT_DIR / "round_template.yaml"
# Round 1 of each line: 261103's weight-1.0 KL-0.01 run of that seed (run key, W&B run id)
ROUND1_FOLDER = S.SYCOPHANCY_DIR / "261103_kl_0_01" / "weight_1_0"
ROUND1_GROUP = "261103_kl_0_01"
ROUND1_RUNS = {42: ("kl_weight_1_0_s42", "am1g59lu"), 43: ("kl_weight_1_0_s43", "hxpczsle")}
DETECTOR_KEY = check_probe_scoring.DETECTOR_KEY
EOT = "<|eot_id|>"
# The per-round question files (labels command)
LABELS_DIR = PARENT_DIR / "labels"
REDRAWN_ROUNDS = (2, 3, 4, 5)
REDRAW_SEED = 261_108  # round R's flipped set: random.Random(REDRAW_SEED + R).sample(sorted train ids, N_FLIPPED_TRAIN)
ORIGINAL_FLIP_SEED = 261_013  # the original train draw (261013 noisy_spec.FLIP_SEED + SPLIT_SEED_OFFSET["train"])
N_FLIPPED_TRAIN = 243  # round(0.2 * 1217), as the original
ORIGINAL_MD5 = "226ed577ee96729ad02a47dbee2d918f"  # md5 of S.QUESTIONS_PATH
LABEL_FIELDS = ("train_label", "label_flipped", "assigned_cue_letter", "assigned_condition")  # the fields a redraw sets


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    print(f"Wrote {path}")


def run_key(round_number: int) -> str:
    assert round_number >= 2, "round 1 is 261103's run (ROUND1_RUNS)"
    return f"s{SEED}_redrawn_round{round_number}"


def label_file(round_number: int) -> Path:
    return LABELS_DIR / f"round{round_number}" / "labels.jsonl"


def run_dir(round_number: int) -> Path:
    if round_number == 1:
        key, expected_run_id = ROUND1_RUNS[SEED]
        run_id = json.loads((ROUND1_FOLDER / f"run_metrics_{key}.json").read_text())["run_id"]
        assert run_id == expected_run_id, (run_id, expected_run_id)
        path = RUN_RESULTS / ROUND1_GROUP / f"{key}_{run_id}"
    else:
        run_id = json.loads((EXP_DIR / f"run_metrics_{run_key(round_number)}.json").read_text())["run_id"]
        path = RUN_RESULTS / GROUP / f"{run_key(round_number)}_{run_id}"
    assert path.is_dir(), path
    return path


def adapter_of(round_number: int) -> Path:
    adapter = run_dir(round_number) / "checkpoints" / "final"
    assert (adapter / "adapter_config.json").exists(), adapter
    return adapter


def probe_folder(any_run_dir: Path, subdir: str) -> Path:
    """The folder of a run's saved probe in the trainer's layout (per-layer model.pt / config.json)."""
    folder = any_run_dir / subdir / f"detector_{subdir}__{DETECTOR_KEY}"
    assert sorted(p.name for p in folder.iterdir()) == [f"layer_{layer}" for layer in S.PROBE_LAYERS], folder
    return folder


def run_dir_of_config(config_path: Path) -> Path:
    """The run folder of a launcher config (run_eval.sh's run_dir)."""
    config = yaml.safe_load(config_path.read_text())
    run_id = json.loads((Path(config["output_dir"]) / f"run_metrics_{config['run_key']}.json").read_text())["run_id"]
    path = RUN_RESULTS / config["group"] / f"{config['run_key']}_{run_id}"
    assert path.is_dir(), path
    return path


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #


def other_letter(letter: str) -> str:
    assert letter in ("A", "B"), letter
    return "B" if letter == "A" else "A"


def redrawn_questions(original: list[dict], seed: int) -> list[dict]:
    """The original question rows with train rows' labels redrawn: build_dataset.py:53-77's draw and label fields."""
    train_ids = sorted(q["question_id"] for q in original if q["split"] == "train")
    assert len(train_ids) == S.N_QUESTIONS["train"], len(train_ids)
    flipped = set(random.Random(seed).sample(train_ids, N_FLIPPED_TRAIN))
    rows = []
    for question in original:
        row = dict(question)  # same key order; assigning existing keys keeps their place
        if row["split"] == "train":
            gold = row["answer"]
            assert row["p_cue_label"] == S.CUE_LABEL_PROB, row["question_id"]
            train_label = other_letter(gold) if row["question_id"] in flipped else gold
            cue_letter = train_label if row["cue_u"] < S.CUE_LABEL_PROB else other_letter(train_label)
            row["train_label"] = train_label
            row["label_flipped"] = row["question_id"] in flipped
            row["assigned_cue_letter"] = cue_letter
            row["assigned_condition"] = "cue_gold" if cue_letter == gold else "cue_dist"
        rows.append(row)
    return rows


def as_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)  # build_dataset.py's serialization


def check_redrawn(original: list[dict], rows: list[dict]) -> None:
    assert len(rows) == len(original) == sum(S.N_QUESTIONS.values()), len(rows)
    for question, row in zip(original, rows, strict=True):
        assert list(row) == list(question), row["question_id"]
        assert row["label_flipped"] == (row["train_label"] != row["answer"]), row["question_id"]
        if row["split"] == "eval":
            assert row == question, row["question_id"]
        else:
            assert {k: v for k, v in row.items() if k not in LABEL_FIELDS} == {
                k: v for k, v in question.items() if k not in LABEL_FIELDS}, row["question_id"]
    assert sum(r["label_flipped"] for r in rows if r["split"] == "train") == N_FLIPPED_TRAIN


def flipped_train_ids(rows: list[dict]) -> set[int]:
    return {r["question_id"] for r in rows if r["split"] == "train" and r["label_flipped"]}


def labels(known_answer: bool) -> None:
    original_text = S.QUESTIONS_PATH.read_text()
    assert hashlib.md5(original_text.encode()).hexdigest() == ORIGINAL_MD5, S.QUESTIONS_PATH
    original = [json.loads(line) for line in original_text.splitlines()]
    if known_answer:
        rebuilt = redrawn_questions(original, ORIGINAL_FLIP_SEED)
        check_redrawn(original, rebuilt)
        assert as_jsonl(rebuilt) == original_text, "the builder does not reproduce the original question file"
        print(f"Known answer PASS: seed {ORIGINAL_FLIP_SEED} reproduces {S.QUESTIONS_PATH} byte for byte (md5 {ORIGINAL_MD5})")
        return
    assert not LABELS_DIR.exists(), f"{LABELS_DIR} exists"
    flipped_sets = {"original": flipped_train_ids(original)}
    md5_lines = []
    summary: dict = {"source": str(S.QUESTIONS_PATH), "source_md5": ORIGINAL_MD5, "n_flipped_train": N_FLIPPED_TRAIN,
                     "seeds": {}, "rounds": {}}
    for round_number in REDRAWN_ROUNDS:
        seed = REDRAW_SEED + round_number
        rows = redrawn_questions(original, seed)
        check_redrawn(original, rows)
        text = as_jsonl(rows)
        path = label_file(round_number)
        path.parent.mkdir(parents=True)
        path.write_text(text)
        print(f"Wrote {path}")
        md5_lines.append(f"{hashlib.md5(text.encode()).hexdigest()}  {path}")
        flipped = flipped_train_ids(rows)
        flipped_sets[f"round{round_number}"] = flipped
        train_rows = [r for r in rows if r["split"] == "train"]
        summary["seeds"][f"round{round_number}"] = seed
        summary["rounds"][f"round{round_number}"] = {
            "flipped": len(flipped),
            "flipped_by_category": dict(sorted(Counter(r["category"] for r in train_rows if r["label_flipped"]).items())),
            "flipped_with_base_confident_correct": sum(r["label_flipped"] and r["base_confident_correct"] for r in train_rows),
            "flipped_gold_first": sum(r["label_flipped"] and r["answer"] == "A" for r in train_rows),
        }
    train_rows = [r for r in original if r["split"] == "train"]
    summary["original"] = {
        "flipped": len(flipped_sets["original"]),
        "flipped_by_category": dict(sorted(Counter(r["category"] for r in train_rows if r["label_flipped"]).items())),
        "flipped_with_base_confident_correct": sum(r["label_flipped"] and r["base_confident_correct"] for r in train_rows),
        "flipped_gold_first": sum(r["label_flipped"] and r["answer"] == "A" for r in train_rows),
        "train_questions_by_category": dict(sorted(Counter(r["category"] for r in train_rows).items())),
    }
    names = list(flipped_sets)
    summary["overlap"] = {f"{a} & {b}": len(flipped_sets[a] & flipped_sets[b])
                          for i, a in enumerate(names) for b in names[i + 1:]}
    (LABELS_DIR / "md5.txt").write_text("\n".join(md5_lines) + "\n")
    write_json(LABELS_DIR / "summary.json", summary)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def config(round_number: int) -> None:
    template = yaml.safe_load(TEMPLATE_PATH.read_text())
    assert template["group"] == GROUP, template["group"]
    checkpoint = str(adapter_of(round_number - 1))
    frozen_probe = str(probe_folder(run_dir(1), S.START_PROBE_SUBDIR))
    questions = label_file(round_number)
    assert questions.is_file(), questions
    for smoke in (False, True) if round_number == 2 else (False,):
        key = f"smoke_{run_key(round_number)}" if smoke else run_key(round_number)
        run_config = copy.deepcopy(template)
        run_config["run_key"] = key
        run_config["output_dir"] = str(EXP_DIR)
        run_config["probe"]["directory_save_prefix"] = key
        run_config["checkpoint"] = checkpoint
        overrides = run_config["overrides"]
        overrides["training.seed"] = SEED
        overrides["training.max_steps"] = 2 if smoke else 128
        overrides["detector.frozen_probe_weights_dir"] = frozen_probe
        overrides["data.task_dataset.mmlu_questions_jsonl"] = str(questions)
        if smoke:
            overrides["training.eval_steps"] = 1
        text = yaml.safe_dump(run_config, sort_keys=False)
        assert "FILL_IN" not in text
        path = EXP_DIR / f"{key}.yaml"
        assert not path.exists(), f"{path} exists"
        path.write_text(f"# Round {round_number} of the seed-{SEED} line{' (smoke run)' if smoke else ''}: written by "
                        f"`rounds.py {EXP_DIR.name} config {round_number}` from ../round_template.yaml.\n"
                        f"#\n#   bash /workspace/sycophancy-rl/sycophancy_setup/run_rl.sh {EXP_DIR} {key} <attempt>\n\n" + text)
        print(f"Wrote {path}")


# --------------------------------------------------------------------------- #
# links
# --------------------------------------------------------------------------- #


def links(round_number: int) -> None:
    folder = EXP_DIR / "links" / f"round{round_number}"
    assert not folder.exists(), f"{folder} exists"
    folder.mkdir(parents=True)
    (folder / S.START_PROBE_SUBDIR).symlink_to(run_dir(1) / S.START_PROBE_SUBDIR)
    (folder / S.FRESH_PROBE_SUBDIR).symlink_to(run_dir(round_number) / S.FRESH_PROBE_SUBDIR)
    for subdir in (S.START_PROBE_SUBDIR, S.FRESH_PROBE_SUBDIR):
        print(f"{folder / subdir} -> {probe_folder(folder, subdir).resolve()}")


# --------------------------------------------------------------------------- #
# penalty_check
# --------------------------------------------------------------------------- #


def difference_summary(new: list[float], saved: list[float]) -> dict:
    differences = [abs(a - b) for a, b in zip(new, saved, strict=True)]
    return {"max_abs": max(differences), "median_abs": statistics.median(differences), "mean_abs": statistics.mean(differences),
            "over_0_02": sum(d > 0.02 for d in differences)}


def penalty_check(config_path: Path, penalty_probe_run: Path, other_probe_run: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    records_path = run_dir_of_config(config_path) / "checkpoints" / "responses" / "step_0.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    scored = [r for r in records if r["probe_score"] is not None]
    assert all(r["parsed_answer"] is not None for r in scored)
    items = []
    for r in scored:
        completion = r["completion"].removesuffix(EOT)
        letter, _, letter_index = find_answer_span(completion)
        assert letter == r["parsed_answer"], (r["question_id"], letter, r["parsed_answer"])
        items.append({"text": r["prompt"] + completion, "letter_char": len(r["prompt"]) + letter_index})
    adapter = Path(config["checkpoint"])
    model, tokenizer = eval_checkpoint.load_hf_model(adapter)
    probes = {"penalty_probe": eval_checkpoint.load_probe(penalty_probe_run, S.START_PROBE_SUBDIR),
              "other_probe": eval_checkpoint.load_probe(other_probe_run, S.START_PROBE_SUBDIR)}
    results = eval_checkpoint.score_followups(model, tokenizer, items, probes)
    saved = [r["probe_score"] for r in scored]
    write_json(EXP_DIR / "checks" / f"penalty_{config['run_key']}.json", {
        "run_config": str(config_path),
        "records": str(records_path),
        "adapter": str(adapter),
        "answers": {"step_0": len(records), "with_probe_score": len(scored)},
        "penalty_probe": str(probe_folder(penalty_probe_run, S.START_PROBE_SUBDIR).resolve()),
        "other_probe": str(probe_folder(other_probe_run, S.START_PROBE_SUBDIR).resolve()),
        "penalty_probe_vs_saved": difference_summary([p["penalty_probe"] for _, _, _, p in results], saved),
        "other_probe_vs_saved": difference_summary([p["other_probe"] for _, _, _, p in results], saved),
    })


def main() -> None:
    global EXP_DIR, SEED
    if sys.argv[1:2] == ["labels"]:  # both lines' question files; no LINE argument
        labels_parser = argparse.ArgumentParser(prog="rounds.py labels", description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter)
        labels_parser.add_argument("--known-answer", action="store_true")
        labels(labels_parser.parse_args(sys.argv[2:]).known_answer)
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("line", choices=tuple(LINE_SEEDS))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("config", "links"):
        commands.add_parser(name).add_argument("round", type=int)
    check_parser = commands.add_parser("penalty_check")
    for name in ("run_yaml", "penalty_probe_run", "other_probe_run"):
        check_parser.add_argument(name, type=Path)
    args = parser.parse_args()
    EXP_DIR = PARENT_DIR / args.line
    SEED = LINE_SEEDS[args.line]
    assert EXP_DIR.is_dir(), EXP_DIR
    print(f"=== rounds.py {args.line} {args.command} ===", flush=True)
    if args.command == "penalty_check":
        penalty_check(args.run_yaml, args.penalty_probe_run, args.other_probe_run)
    else:
        assert args.round in ROUNDS and args.round >= 2, args.round
        {"config": config, "links": links}[args.command](args.round)


if __name__ == "__main__":
    main()
