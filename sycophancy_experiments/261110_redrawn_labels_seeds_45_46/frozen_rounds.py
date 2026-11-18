"""The frozen version's steps around the RL runs of seeds 45 and 46 with the training labels redrawn every round: per
line (seed), rounds 2-5 continuing this folder's round 1 (LINE/s<seed>_round1.yaml, the same round 1 the refit version
in rounds.py continues), against round 1's probe in every round (never refit), round R training on 261108's question
file for that round (261108_frozen_probe_redrawn_labels/labels/roundR/labels.jsonl). 261108's rounds.py with round 1
read from this folder, the label files read from 261108 (no labels command), and no smoke run.

    source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh && cd <this folder>
    $PY frozen_rounds.py LINE config R          -> LINE/s<seed>_frozen_roundR.yaml (R >= 2)
    $PY frozen_rounds.py LINE links R           -> LINE/links/roundR/
    $PY frozen_rounds.py LINE penalty_check RUN_YAML PENALTY_PROBE_RUN OTHER_PROBE_RUN   -> LINE/checks/penalty_<run key>.json

LINE is seed_45 or seed_46: the line's folder, under this one.

Round 1 of a line is LINE/s<seed>_round1.yaml's run: its folder is
/workspace/results/run_results/<this folder's name>/s<seed>_round1_<run id> (run id in LINE/run_metrics_s<seed>_round1.json).
Its start-of-run probe (fit on the base model's answers through the base model) is the line's frozen probe. Frozen round
R >= 2's run is LINE/s<seed>_frozen_roundR.yaml; its folder is
/workspace/results/run_results/<this folder's name>/s<seed>_frozen_roundR_<run id> (run id in
LINE/run_metrics_s<seed>_frozen_roundR.json), and its final adapter that folder's checkpoints/final.

The trainer still fits its own start-of-run probe (on round 1's probe files, through round R-1's model) and saves it in
the run folder's start-of-run slot, then swaps in the frozen probe for the penalty
(train_grpo_model_against_probe.py:553-575). So the run folder's start-of-run slot holds a probe the run never used.

config          s<seed>_frozen_roundR.yaml (R >= 2) from frozen_round_template.yaml: the line's seed and folder; round
                R-1's final adapter as the checkpoint (round 2: round 1's; R >= 3: frozen round R-1's);
                detector.frozen_probe_weights_dir = round 1's start-of-run probe folder;
                data.task_dataset.mmlu_questions_jsonl = 261108's labels/roundR/labels.jsonl, whose md5 must equal
                261108's labels/md5.txt.
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
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "/workspace/sycophancy-rl/sycophancy_setup")
import check_probe_scoring  # noqa: E402
import eval_checkpoint  # noqa: E402
import spec as S  # noqa: E402
from obfuscation_atlas.tasks.multiple_choice import find_answer_span  # noqa: E402

PARENT_DIR = Path(__file__).resolve().parent
GROUP = PARENT_DIR.name  # W&B group and /workspace/results/run_results/<group>/
LINE_SEEDS = {"seed_45": 45, "seed_46": 46}
EXP_DIR: Path  # the line's folder, PARENT_DIR / LINE; set by main from the line argument
SEED: int  # the line's training seed; set by main
RUN_RESULTS = Path("/workspace/results/run_results")
ROUNDS = (1, 2, 3, 4, 5)
TEMPLATE_PATH = PARENT_DIR / "frozen_round_template.yaml"
DETECTOR_KEY = check_probe_scoring.DETECTOR_KEY
EOT = "<|eot_id|>"
# Round R's question file (training labels redrawn per round), shared with 261108 and 261109
LABELS_DIR = S.SYCOPHANCY_DIR / "261108_frozen_probe_redrawn_labels" / "labels"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    print(f"Wrote {path}")


def run_key(round_number: int) -> str:
    assert round_number in ROUNDS, round_number
    return f"s{SEED}_round1" if round_number == 1 else f"s{SEED}_frozen_round{round_number}"


def label_file(round_number: int) -> Path:
    """Round R's question file from 261108, after checking its md5 against 261108's labels/md5.txt."""
    path = LABELS_DIR / f"round{round_number}" / "labels.jsonl"
    recorded = dict(reversed(line.split("  ")) for line in (LABELS_DIR / "md5.txt").read_text().splitlines())
    assert hashlib.md5(path.read_bytes()).hexdigest() == recorded[str(path)], path
    return path


def run_dir(round_number: int) -> Path:
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
# config
# --------------------------------------------------------------------------- #


def config(round_number: int) -> None:
    template = yaml.safe_load(TEMPLATE_PATH.read_text())
    assert template["group"] == GROUP, template["group"]
    checkpoint = str(adapter_of(round_number - 1))
    frozen_probe = str(probe_folder(run_dir(1), S.START_PROBE_SUBDIR))
    questions = label_file(round_number)
    key = run_key(round_number)
    run_config = copy.deepcopy(template)
    run_config["run_key"] = key
    run_config["output_dir"] = str(EXP_DIR)
    run_config["probe"]["directory_save_prefix"] = key
    run_config["checkpoint"] = checkpoint
    overrides = run_config["overrides"]
    overrides["training.seed"] = SEED
    overrides["training.max_steps"] = 128
    overrides["detector.frozen_probe_weights_dir"] = frozen_probe
    overrides["data.task_dataset.mmlu_questions_jsonl"] = str(questions)
    text = yaml.safe_dump(run_config, sort_keys=False)
    assert "FILL_IN" not in text
    path = EXP_DIR / f"{key}.yaml"
    assert not path.exists(), f"{path} exists"
    path.write_text(f"# Frozen round {round_number} of the seed-{SEED} line: written by "
                    f"`frozen_rounds.py {EXP_DIR.name} config {round_number}` from ../frozen_round_template.yaml.\n"
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
    print(f"=== frozen_rounds.py {args.line} {args.command} ===", flush=True)
    if args.command == "penalty_check":
        penalty_check(args.run_yaml, args.penalty_probe_run, args.other_probe_run)
    else:
        assert args.round in ROUNDS and args.round >= 2, args.round
        {"config": config, "links": links}[args.command](args.round)


if __name__ == "__main__":
    main()
