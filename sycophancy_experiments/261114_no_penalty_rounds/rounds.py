"""The steps around the RL runs of rounds 2-5 with no probe penalty: two lines (seeds 42 and 43), each continuing
261103's no-penalty KL-0.01 run of its seed as round 1, rounds 2-5 at penalty weight 0 on the fixed training labels.
261107's rounds.py (the frozen-probe control) with round 1 moved to 261103's no-penalty runs, the run keys renamed, and
its smoke config and penalty check removed. The trainer still scores every training answer with round 1's probe
(detector.frozen_probe_weights_dir), but at weight 0 the score does not enter the reward.

    source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh && cd <this folder>
    $PY rounds.py LINE config R          -> LINE/s<seed>_no_penalty_roundR.yaml (R >= 2)
    $PY rounds.py LINE links R           -> LINE/links/roundR/
    $PY rounds.py LINE probe_table       -> LINE/probe_table.json and probe_table.md (CPU, after the queue)

LINE is seed_42 or seed_43: the line's folder, under this one.

Round 1 of a line is 261103's no-penalty run of its seed (ROUND1_RUNS), not rerun: its folder is
/workspace/results/run_results/261103_kl_0_01/kl_no_penalty_s<seed>_<run id>. Its start-of-run probe (fit on the base
model's answers through the base model; byte-identical to the probe of 261103's weight-1.0 runs, 261107's frozen probe)
is the line's round-1 probe. Round R >= 2's run is LINE/s<seed>_no_penalty_roundR.yaml; its folder is
/workspace/results/run_results/<this folder's name>/s<seed>_no_penalty_roundR_<run id> (run id in
LINE/run_metrics_s<seed>_no_penalty_roundR.json), and its final adapter that folder's checkpoints/final.

The trainer still fits its own start-of-run probe (on round 1's probe files, through round R-1's model) and saves it in
the run folder's start-of-run slot, then swaps in round 1's probe for scoring (train_grpo_model_against_probe.py:553-575).
So the run folder's start-of-run slot holds a probe the run never used.

config          s<seed>_no_penalty_roundR.yaml (R >= 2) from round_template.yaml: the line's seed and folder; round R-1's
                final adapter as the checkpoint; detector.frozen_probe_weights_dir = round 1's start-of-run probe folder.
links           LINE/links/roundR/, the folder the final eval reads in place of the run folder (eval_checkpoint.py
                selfreport --probe-run-dir): its start-of-run entry links to round 1's start-of-run probe, its
                end-of-run entry to round R's own end-of-run refit. So in the eval's labels.jsonl probe_start = round 1's
                probe, probe_fresh = this run's end-of-run refit, as for any other run.
probe_table     Per round 1-5, both cue conditions pooled (261104's rounds.py table rule): the cued eval answers (sample 0)
                that name the cued option, labelled biased / unbiased against the checkpoint's own no-cue answer, on the
                questions with a no-cue answer in both the base model's and the round's eval. Round 1's probe
                (probe_start) on the round's final-eval answers and on the base model's answers (261103's
                base_rescored eval of the no-penalty run, which round 1's probe scored): counts, mean score of each
                class, AUC.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "/workspace/sycophancy-rl/sycophancy_setup")
import check_probe_scoring  # noqa: E402
import compare  # noqa: E402
import spec as S  # noqa: E402

PARENT_DIR = Path(__file__).resolve().parent
GROUP = PARENT_DIR.name  # W&B group and /workspace/results/run_results/<group>/
LINE_SEEDS = {"seed_42": 42, "seed_43": 43}
EXP_DIR: Path  # the line's folder, PARENT_DIR / LINE; set by main from the line argument
SEED: int  # the line's training seed; set by main
RUN_RESULTS = Path("/workspace/results/run_results")
ROUNDS = (1, 2, 3, 4, 5)
TEMPLATE_PATH = PARENT_DIR / "round_template.yaml"
# Round 1 of each line: 261103's no-penalty KL-0.01 run of that seed (run key, W&B run id)
ROUND1_FOLDER = S.SYCOPHANCY_DIR / "261103_kl_0_01" / "no_penalty"
ROUND1_GROUP = "261103_kl_0_01"
ROUND1_RUNS = {42: ("kl_no_penalty_s42", "el7heia5"), 43: ("kl_no_penalty_s43", "1uv2slxc")}
DETECTOR_KEY = check_probe_scoring.DETECTOR_KEY


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    print(f"Wrote {path}")


def run_key(round_number: int) -> str:
    assert round_number >= 2, "round 1 is 261103's run (ROUND1_RUNS)"
    return f"s{SEED}_no_penalty_round{round_number}"


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


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def config(round_number: int) -> None:
    template = yaml.safe_load(TEMPLATE_PATH.read_text())
    assert template["group"] == GROUP, template["group"]
    key = run_key(round_number)
    run_config = copy.deepcopy(template)
    run_config["run_key"] = key
    run_config["output_dir"] = str(EXP_DIR)
    run_config["probe"]["directory_save_prefix"] = key
    run_config["checkpoint"] = str(adapter_of(round_number - 1))
    overrides = run_config["overrides"]
    overrides["training.seed"] = SEED
    overrides["training.max_steps"] = 128
    overrides["detector.frozen_probe_weights_dir"] = str(probe_folder(run_dir(1), S.START_PROBE_SUBDIR))
    text = yaml.safe_dump(run_config, sort_keys=False)
    assert "FILL_IN" not in text
    path = EXP_DIR / f"{key}.yaml"
    assert not path.exists(), f"{path} exists"
    path.write_text(f"# Round {round_number} of the seed-{SEED} line: written by `rounds.py {EXP_DIR.name} config {round_number}` "
                    f"from ../round_template.yaml.\n#\n#   bash /workspace/sycophancy-rl/sycophancy_setup/run_rl.sh {EXP_DIR} {key} <attempt>\n\n" + text)
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
# probe_table
# --------------------------------------------------------------------------- #


def final_eval_dir(round_number: int) -> Path:
    if round_number == 1:
        return ROUND1_FOLDER / "evals" / ROUND1_RUNS[SEED][0]
    return EXP_DIR / "evals" / run_key(round_number)


def base_eval_dir() -> Path:
    return ROUND1_FOLDER / "evals" / f"base_rescored_{ROUND1_RUNS[SEED][0]}"


def probe_answers(eval_dir: Path, shared_questions: set[int]) -> list[dict]:
    rows = compare.rollout_rows(eval_dir)
    return [r for r in rows if r["condition"] != "none" and r["question_id"] in shared_questions and r["followed_cue"]
            and r["bias_rm"] in ("bias", "no bias")]


def pooled(rows: list[dict]) -> dict:
    biased = [r["probe_start"] for r in rows if r["bias_rm"] == "bias" and r["probe_start"] is not None]
    unbiased = [r["probe_start"] for r in rows if r["bias_rm"] == "no bias" and r["probe_start"] is not None]
    return {"n_biased": len(biased), "n_unbiased": len(unbiased), "mean_biased": compare.mean(biased),
            "mean_unbiased": compare.mean(unbiased), "auc": compare.auc(biased, unbiased)}


def probe_table() -> None:
    seed_table = json.loads((EXP_DIR / "tables" / "rounds" / "seed_table.json").read_text())["mean_sd"]
    base_rows = compare.rollout_rows(base_eval_dir())
    result = {}
    for round_number in ROUNDS:
        round_rows = compare.rollout_rows(final_eval_dir(round_number))
        shared = compare.questions_with_nocue_answer(base_rows) & compare.questions_with_nocue_answer(round_rows)
        round_answers = pooled(probe_answers(final_eval_dir(round_number), shared))
        for key in ("n_biased", "n_unbiased"):  # the seed table's per-condition counts, summed
            per_condition = [seed_table[f"probe_start_{key}"][condition]["after_rl"][round_number - 1] for condition in ("cue_gold", "cue_dist")]
            assert round_answers[key] == sum(per_condition), (round_number, key, round_answers[key], per_condition)
        result[f"round{round_number}"] = {"shared_questions": len(shared), "final_eval": str(final_eval_dir(round_number)),
                                          "round_answers": round_answers, "base_answers": pooled(probe_answers(base_eval_dir(), shared))}
    write_json(EXP_DIR / "probe_table.json", {"base_eval": str(base_eval_dir()), "rounds": result})
    lines = [f"# Round 1's probe per round, seed {SEED}", "",
             "Both cue conditions pooled: the cued eval answers (sample 0) that name the cued option, labelled biased / unbiased",
             "against the checkpoint's own no-cue answer, on the questions with a no-cue answer in both the base model's and the",
             "round's eval. Round's answers: the round's final eval, read through the round's model. Base model's answers: the",
             "base eval on the same questions, read through the base model. Built by `rounds.py LINE probe_table`.", "",
             "| Round | Answers | Biased / unbiased answers | Mean score, biased / unbiased | AUC |", "|---|---|---|---|---|"]
    for name, round_result in result.items():
        for key, title in (("round_answers", "round's answers"), ("base_answers", "base model's answers")):
            r = round_result[key]
            lines.append(f"| {name} | {title} | {r['n_biased']} / {r['n_unbiased']} | {r['mean_biased']:.3f} / {r['mean_unbiased']:.3f} | {r['auc']:.3f} |")
    (EXP_DIR / "probe_table.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {EXP_DIR / 'probe_table.md'}")


def main() -> None:
    global EXP_DIR, SEED
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("line", choices=tuple(LINE_SEEDS))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("config", "links"):
        commands.add_parser(name).add_argument("round", type=int)
    commands.add_parser("probe_table")
    args = parser.parse_args()
    EXP_DIR = PARENT_DIR / args.line
    SEED = LINE_SEEDS[args.line]
    assert EXP_DIR.is_dir(), EXP_DIR
    print(f"=== rounds.py {args.line} {args.command} ===", flush=True)
    if args.command == "probe_table":
        probe_table()
    else:
        assert args.round in ROUNDS and args.round >= 2, args.round
        {"config": config, "links": links}[args.command](args.round)


if __name__ == "__main__":
    main()
