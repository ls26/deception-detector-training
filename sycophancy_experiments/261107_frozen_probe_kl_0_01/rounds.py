"""The steps around the RL runs of the frozen-probe control for 261104: two lines (seeds 42 and 43), each continuing
261103's weight-1.0 run of its seed as round 1, rounds 2-5 against round 1's probe (never refit). 261104's rounds.py
cut down to the three steps a frozen probe needs.

    source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh && cd <this folder>
    $PY rounds.py LINE config R          -> LINE/s<seed>_frozen_roundR.yaml (R >= 2; round 2 also LINE/smoke_s<seed>_frozen_round2.yaml)
    $PY rounds.py LINE links R           -> LINE/links/roundR/
    $PY rounds.py LINE penalty_check RUN_YAML PENALTY_PROBE_RUN OTHER_PROBE_RUN   -> LINE/checks/penalty_<run key>.json
    $PY rounds.py LINE probe_table       -> LINE/probe_table.json and probe_table.md (CPU, after the queue)

LINE is seed_42 or seed_43: the line's folder, under this one.

Round 1 of a line is 261103's weight-1.0 run of its seed (ROUND1_RUNS), not rerun: its folder is
/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s<seed>_<run id>. Its start-of-run probe (fit on the base
model's answers through the base model) is the line's frozen probe. Round R >= 2's run is LINE/s<seed>_frozen_roundR.yaml;
its folder is /workspace/results/run_results/<this folder's name>/s<seed>_frozen_roundR_<run id> (run id in
LINE/run_metrics_s<seed>_frozen_roundR.json), and its final adapter that folder's checkpoints/final.

The trainer still fits its own start-of-run probe (on round 1's probe files, through round R-1's model) and saves it in
the run folder's start-of-run slot, then swaps in the frozen probe for the penalty
(train_grpo_model_against_probe.py:553-575). So the run folder's start-of-run slot holds a probe the run never used.

config          s<seed>_frozen_roundR.yaml (R >= 2) from round_template.yaml: the line's seed and folder; round R-1's final
                adapter as the checkpoint; detector.frozen_probe_weights_dir = round 1's start-of-run probe folder.
                smoke_s<seed>_frozen_round2.yaml (round 2): the same with 2 steps and an in-loop eval after each.
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
probe_table     Per round 1-5, both cue conditions pooled (261104's rounds.py table rule): the cued eval answers (sample 0)
                that name the cued option, labelled biased / unbiased against the checkpoint's own no-cue answer, on the
                questions with a no-cue answer in both the base model's and the round's eval. Round 1's probe
                (probe_start) on the round's final-eval answers and on the base model's answers (261103's
                base_rescored eval, which round 1's probe scored): counts, mean score of each class, AUC.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "/workspace/sycophancy-rl/sycophancy_setup")
import check_probe_scoring  # noqa: E402
import compare  # noqa: E402
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


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    print(f"Wrote {path}")


def run_key(round_number: int) -> str:
    assert round_number >= 2, "round 1 is 261103's run (ROUND1_RUNS)"
    return f"s{SEED}_frozen_round{round_number}"


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
# config
# --------------------------------------------------------------------------- #


def config(round_number: int) -> None:
    template = yaml.safe_load(TEMPLATE_PATH.read_text())
    assert template["group"] == GROUP, template["group"]
    checkpoint = str(adapter_of(round_number - 1))
    frozen_probe = str(probe_folder(run_dir(1), S.START_PROBE_SUBDIR))
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
    check_parser = commands.add_parser("penalty_check")
    for name in ("run_yaml", "penalty_probe_run", "other_probe_run"):
        check_parser.add_argument(name, type=Path)
    commands.add_parser("probe_table")
    args = parser.parse_args()
    EXP_DIR = PARENT_DIR / args.line
    SEED = LINE_SEEDS[args.line]
    assert EXP_DIR.is_dir(), EXP_DIR
    print(f"=== rounds.py {args.line} {args.command} ===", flush=True)
    if args.command == "penalty_check":
        penalty_check(args.run_yaml, args.penalty_probe_run, args.other_probe_run)
    elif args.command == "probe_table":
        probe_table()
    else:
        assert args.round in ROUNDS and args.round >= 2, args.round
        {"config": config, "links": links}[args.command](args.round)


if __name__ == "__main__":
    main()
