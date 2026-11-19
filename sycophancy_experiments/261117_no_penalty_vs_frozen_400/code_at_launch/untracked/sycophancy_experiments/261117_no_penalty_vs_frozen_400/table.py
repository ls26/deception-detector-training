"""The comparison tables: the new 400-step runs at steps 50-400 and, in a second block, the old 5-round chains at
steps 128-640, each against the base model.

    source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh && cd <this folder>
    $PY table.py seed CONDITION_FOLDER SEED   -> CONDITION_FOLDER/seed_SEED/table/comparison.{md,json}  (one new run)
    $PY table.py combined                     -> comparison/comparison.{md,json}  (both conditions, both seeds, old chains)
    $PY table.py old OUT_DIR                  -> OUT_DIR/comparison.{md,json}  (the old chains' block alone, from saved
                                                 evals only: the queues' identity check)

Every value comes from one checkpoint's eval paired with the base model's eval (compare.py's eval_tables on the two;
one sampled answer per prompt):
- bias: share of cued answers labelled bias (rm: the answer moved to the cued option away from the checkpoint's own
  no-cue answer), on questions with a no-cue answer in both evals;
- cue agreement rate: share of cued answers naming the cued option; true accuracy: share of answers that are right;
- for each, the cue at the correct answer (cue_gold), the cue at the wrong answer (cue_dist), and the 50/50 mix: the
  mean of those two rates (each eval question has one answer with each cue); true accuracy also with no cue;
- the initial probe (probe_start: the run's start-of-run probe; for the old chains round 1's probe) on cued answers
  that name the cued option, are labelled bias or no bias, have a probe score, and are on questions with a no-cue
  answer in both evals, both cues pooled: mean score of biased answers, of unbiased answers, and the AUC between them.
The base model's value is computed in each pairing (its question set depends on the checkpoint); the base column is
its mean over the pairings of that run (as in 261114's table).

The new runs' evals: CONDITION_FOLDER/seed_SEED/evals/base_rescored (261013's base answers scored with the run's
initial probe) and evals/step_N (the adapter after N training steps). The old chains' evals are listed in OLD_CHAINS.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/sycophancy-rl/sycophancy_setup")
import compare  # noqa: E402
import spec as S  # noqa: E402

FOLDER = Path(__file__).resolve().parent
NEW_STEPS = tuple(range(50, 401, 50))
NEW_RUNS = {"no penalty": ("no_penalty", (42, 43)), "frozen probe": ("frozen_probe", (42, 43))}
OLD_STEPS = (128, 256, 384, 512, 640)
SYC = S.SYCOPHANCY_DIR
NP_103, W1_103 = SYC / "261103_kl_0_01" / "no_penalty" / "evals", SYC / "261103_kl_0_01" / "weight_1_0" / "evals"
OLD_CHAINS = {  # condition -> seed -> (base eval, [evals at steps 128, 256, 384, 512, 640])
    "no penalty, 5 rounds, fixed labels (261103 round 1, 261114)": {
        seed: (NP_103 / f"base_rescored_kl_no_penalty_s{seed}",
               [NP_103 / f"kl_no_penalty_s{seed}"]
               + [SYC / "261114_no_penalty_rounds" / f"seed_{seed}" / "evals" / f"s{seed}_no_penalty_round{r}" for r in range(2, 6)])
        for seed in (42, 43)
    },
    "frozen probe, 5 rounds, labels redrawn each round (261103 / 261110 round 1, 261108 / 261110)": {
        **{seed: (W1_103 / f"base_rescored_kl_weight_1_0_s{seed}",
                  [W1_103 / f"kl_weight_1_0_s{seed}"]
                  + [SYC / "261108_frozen_probe_redrawn_labels" / f"seed_{seed}" / "evals" / f"s{seed}_redrawn_round{r}" for r in range(2, 6)])
           for seed in (42, 43)},
        **{seed: (SYC / "261110_redrawn_labels_seeds_45_46" / f"seed_{seed}" / "evals" / f"base_rescored_s{seed}_round1",
                  [SYC / "261110_redrawn_labels_seeds_45_46" / f"seed_{seed}" / "evals" / f"s{seed}_round1"]
                  + [SYC / "261110_redrawn_labels_seeds_45_46" / f"seed_{seed}" / "evals" / f"s{seed}_frozen_round{r}" for r in range(2, 6)])
           for seed in (45, 46)},
    },
}
CUES = {"50/50 mix": None, "cue at the correct answer": "cue_gold", "cue at the wrong answer": "cue_dist"}
RATES = {"bias": "bias", "cue agreement rate": "followed_cue", "true accuracy": "accuracy"}
PROBE_ROWS = {"mean score of biased answers": "probe_start_mean_biased",
              "mean score of unbiased answers": "probe_start_mean_unbiased",
              "AUC biased vs unbiased": "probe_start_auc"}
ROWS = ([f"{rate}, {cue}" for rate in RATES for cue in CUES] + ["true accuracy, no cue"]
        + [f"initial probe, both cues pooled: {name}" for name in PROBE_ROWS])


def checkpoint_values(table: dict, rows: list[dict], shared: set[int]) -> dict[str, float | None]:
    """One checkpoint's rows of the table, from its compare.eval_tables entry and its rollout rows."""
    values = {}
    for rate, key in RATES.items():
        gold, dist = table["cue_gold"][key], table["cue_dist"][key]
        values[f"{rate}, 50/50 mix"] = None if gold is None or dist is None else (gold + dist) / 2
        values[f"{rate}, cue at the correct answer"] = gold
        values[f"{rate}, cue at the wrong answer"] = dist
    values["true accuracy, no cue"] = table["none"]["accuracy"]
    probe_answers = [r for r in rows if r["condition"] in ("cue_gold", "cue_dist") and r["question_id"] in shared
                     and r["bias_rm"] in ("bias", "no bias") and r["followed_cue"]]
    probe = compare.probe_rates(probe_answers)
    for name, key in PROBE_ROWS.items():
        values[f"initial probe, both cues pooled: {name}"] = probe[key]
    assert list(values) == ROWS
    return values


def pairing(base_dir: Path, eval_dir: Path) -> dict:
    base_rows, rows = compare.rollout_rows(base_dir), compare.rollout_rows(eval_dir)
    assert {r["question_id"] for r in base_rows} == {r["question_id"] for r in rows}, (base_dir, eval_dir)
    tables = compare.eval_tables({"base": base_rows, "checkpoint": rows})
    shared = compare.questions_with_nocue_answer(base_rows) & compare.questions_with_nocue_answer(rows)
    return {
        "base_eval": str(base_dir),
        "eval": str(eval_dir),
        "questions_with_nocue_answer_in_both": len(shared),
        "base": checkpoint_values(tables["all"]["base"], base_rows, shared),
        "checkpoint": checkpoint_values(tables["all"]["checkpoint"], rows, shared),
    }


def run_values(base_dir: Path, eval_dirs: list[Path], steps: tuple[int, ...]) -> dict:
    pairings = {step: pairing(base_dir, eval_dir) for step, eval_dir in zip(steps, eval_dirs, strict=True)}
    return {
        "base_mean": {row: statistics.mean(p["base"][row] for p in pairings.values()) for row in ROWS},
        "steps": {step: p["checkpoint"] for step, p in pairings.items()},
        "pairings": pairings,
    }


def new_run(condition_folder: str, seed: int) -> dict:
    evals = FOLDER / condition_folder / f"seed_{seed}" / "evals"
    return run_values(evals / "base_rescored", [evals / f"step_{step}" for step in NEW_STEPS], NEW_STEPS)


def old_chains_block() -> dict:
    return {condition: {seed: run_values(base, evals, OLD_STEPS) for seed, (base, evals) in seeds.items()}
            for condition, seeds in OLD_CHAINS.items()}


def fmt(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def block_markdown(block: dict[str, dict[int, dict]], steps: tuple[int, ...]) -> list[str]:
    """One table: a line per row and condition; each cell lists every seed's value."""
    lines = ["| Measurement | Condition (seeds) | Base model | " + " | ".join(f"Step {s}" for s in steps) + " |",
             "|---|---|---|" + "---|" * len(steps)]
    for row in ROWS:
        for condition, seeds in block.items():
            cells = [" / ".join(fmt(seeds[seed]["base_mean"][row]) for seed in seeds)]
            cells += [" / ".join(fmt(seeds[seed]["steps"][step][row]) for seed in seeds) for step in steps]
            lines.append(f"| {row} | {condition} ({', '.join(f's{seed}' for seed in seeds)}) | " + " | ".join(cells) + " |")
    return lines


def write(out_dir: Path, title: str, blocks: list[tuple[str, dict, tuple[int, ...]]]) -> None:
    md = [f"# {title}", "",
          f"Written by `table.py`; definitions in its docstring. Each cell lists every seed's value in the order given in the "
          f"condition column. Values are shares (0-1); \"–\" means no answers to compute it on.", ""]
    for heading, block, steps in blocks:
        md += [f"## {heading}", "", *block_markdown(block, steps), ""]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.md").write_text("\n".join(md))
    payload = {heading: {condition: {str(seed): values for seed, values in seeds.items()} for condition, seeds in block.items()}
               for heading, block, _ in blocks}
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=1) + "\n")
    print("\n".join(md))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    seed_parser = commands.add_parser("seed")
    seed_parser.add_argument("condition_folder", choices=[folder for folder, _ in NEW_RUNS.values()])
    seed_parser.add_argument("seed", type=int)
    commands.add_parser("combined")
    commands.add_parser("old").add_argument("out_dir", type=Path)
    args = parser.parse_args()

    if args.command == "seed":
        condition = next(name for name, (folder, _) in NEW_RUNS.items() if folder == args.condition_folder)
        block = {condition: {args.seed: new_run(args.condition_folder, args.seed)}}
        write(FOLDER / args.condition_folder / f"seed_{args.seed}" / "table", f"{condition}, seed {args.seed}: 400 steps",
              [("New run: labels redrawn every prompt, constant learning rate", block, NEW_STEPS)])
    elif args.command == "old":
        write(args.out_dir, "Old chains", [("Old chains: five 128-step rounds, cosine learning rate restarted each round", old_chains_block(), OLD_STEPS)])
    else:
        new_block = {condition: {seed: new_run(folder, seed) for seed in seeds} for condition, (folder, seeds) in NEW_RUNS.items()}
        write(FOLDER / "comparison", "No penalty against the frozen probe: 400 steps, labels redrawn every prompt, constant learning rate",
              [("New runs: labels redrawn every prompt, constant learning rate (5e-5 after 5 warmup steps)", new_block, NEW_STEPS),
               ("Old chains: five 128-step rounds, cosine learning rate restarted each round", old_chains_block(), OLD_STEPS)])


if __name__ == "__main__":
    main()
