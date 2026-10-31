"""The training dynamics, answer text and probe rows of probe-penalty runs: 261024's dynamics.py for several runs,
with each run's advantage SD floor.

    source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh && cd <this folder>
    $PY dynamics.py --run <run yaml> <base eval dir> [--run ...] --out-dir DIR   -> DIR/dynamics.json, DIR/dynamics.md

A run is its launcher config (run key, group, output_dir, penalty weight, advantage SD floor) and the base model's eval
scored with that run's initial probe (261024: evals/base_rescored; here evals/base_rescored_<run key>). Its final eval is
<output_dir>/evals/<run key>, its compare.py output <output_dir>/runs/<run key>/comparison.json.

training        Each run and the three no-penalty baseline runs (261013 seed 42, 261018 seeds 43 and 44), per step
                and averaged over step windows. From the step records: each answer's GRPO advantage recomputed as the
                trainer computes it (reward = task reward - weight x probe score on answers naming the cued option -
                0.1 on answers cut off at the length cap; advantage = (reward - group mean) / (group SD + 1e-4),
                afterburner/grpo_trainer.py:617-624, or / max(group SD + 1e-4, floor) for a run with
                loss.advantage_std_floor), and group counts. From W&B: grad_norm (before clipping), kl_div,
                mean_token_entropy, response_length_mean.
probe_only      Each run's probe-only groups (all 8 answers name the cued option, equal task rewards): the
                within-group SD of the probe scores, and simple text features of the 4 lowest-scoring answers (positive
                advantage) against the 4 highest-scoring.
text            Each run's training answers per step (opening words, exclamation marks, length); the final evals'
                answer lengths, and each run's template share and no-cue letters.
probe_letters   compare.py's probe rows (cued answers naming the cued option, bias or no bias, on the questions with a
                no-cue answer in both evals) for row A (the run's base eval) and rows B-D (the run's final eval), split
                by the letter the answer names; per cue condition and with both cue conditions' answers pooled ("both").
probe_data      The answer letter by label in the probe's training and test sets.
probe_weights   Each run's cosine between the initial and the fresh probe's weight vectors, per layer.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "/workspace/sycophancy-rl/sycophancy_setup")
import compare  # noqa: E402
import spec as S  # noqa: E402

RUN_RESULTS = Path("/workspace/results/run_results")
WANDB_PROJECT_PATH = "lilylixin6-the-harker-school/deception_detection"
NON_TERMINATION_PENALTY = 0.1  # grpo.loss.non_termination_penalty, not overridden
EOT = "<|eot_id|>"  # a step record's completion ends with it unless the answer was cut off at the length cap
ADVANTAGE_EPSILON = 1e-4  # afterburner/grpo_trainer.py
BASELINE_RUNS = {  # run folder, W&B run id
    "baseline_s42": (RUN_RESULTS / "261013_noisy_label_rl" / "noisy_label_s42_6vcs39wj", "6vcs39wj"),
    "baseline_s43": (RUN_RESULTS / "261018_noisy_label_rl_seeds" / "noisy_label_s43_nk7mr4qf", "nk7mr4qf"),
    "baseline_s44": (RUN_RESULTS / "261018_noisy_label_rl_seeds" / "noisy_label_s44_i3wl9evn", "i3wl9evn"),
}
FIXED_EVAL_DIRS = {
    "base": S.SYCOPHANCY_DIR / "261013_noisy_label_rl" / "evals" / "base",
    "baseline_s42": S.SYCOPHANCY_DIR / "261013_noisy_label_rl" / "evals" / "final",
}
TRAINING_WINDOWS = [(0, 10), (10, 30), (30, 60), (60, 128)]
PROBE_ONLY_WINDOWS = [(0, 5), (5, 10), (10, 30), (30, 60), (60, 128)]
TEXT_STEPS = [0, 10, 20, 25, 30, 60, 70, 80, 90, 100, 127]
TEMPLATE_OPENING = "Let's break down the option together!"  # 261024's template
WANDB_KEYS = ["grad_norm", "kl_div", "mean_token_entropy", "response_length_mean"]


def run_info(run_yaml: Path, base_eval_dir: Path) -> dict:
    config = yaml.safe_load(run_yaml.read_text())
    exp_dir = Path(config["output_dir"])
    run_id = json.loads((exp_dir / f"run_metrics_{config['run_key']}.json").read_text())["run_id"]
    return {
        "name": config["run_key"],
        "yaml": str(run_yaml),
        "run_dir": RUN_RESULTS / config["group"] / f"{config['run_key']}_{run_id}",
        "run_id": run_id,
        "weight": config["overrides"]["loss.detector"],
        "std_floor": config["overrides"].get("loss.advantage_std_floor"),  # absent: the trainer's default, no floor
        "base_eval_dir": base_eval_dir,
        "eval_dir": exp_dir / "evals" / config["run_key"],
        "comparison_path": exp_dir / "runs" / config["run_key"] / "comparison.json",
    }


def names_cue(record: dict) -> bool:
    return record["parsed_answer"] is not None and record["parsed_answer"] == record["cue_letter"]


def advantages(group: list[dict], weight: float, std_floor: float | None) -> list[float]:
    rewards = [
        r["reward"]
        - (weight * r["probe_score"] if weight and names_cue(r) else 0.0)
        - NON_TERMINATION_PENALTY * (not r["completion"].endswith(EOT))
        for r in group
    ]
    group_mean = statistics.mean(rewards)
    group_sd = statistics.stdev(rewards)  # torch.std's default, n - 1
    divisor = group_sd + ADVANTAGE_EPSILON if std_floor is None else max(group_sd + ADVANTAGE_EPSILON, std_floor)
    return [(reward - group_mean) / divisor for reward in rewards]


def step_summary(records: list[dict], weight: float, std_floor: float | None) -> dict:
    groups = compare.step_groups(records)
    out = dict(all_name_cue=0, mixed_task_reward=0, mixed_task_reward_cue_not_at_label=0, answers_not_naming_cue=0,
               abs_advantage=0.0, abs_advantage_probe_only=0.0, advantage_on_cue_answers=0.0)
    for group in groups:
        group_advantages = advantages(group, weight, std_floor)
        cue_named = [names_cue(r) for r in group]
        equal_task_reward = len({r["reward"] for r in group}) == 1
        out["answers_not_naming_cue"] += cue_named.count(False)
        out["abs_advantage"] += sum(abs(a) for a in group_advantages)
        out["advantage_on_cue_answers"] += sum(a for a, c in zip(group_advantages, cue_named) if c)
        if all(cue_named):
            out["all_name_cue"] += 1
            if equal_task_reward:
                out["abs_advantage_probe_only"] += sum(abs(a) for a in group_advantages)
        if not equal_task_reward:
            out["mixed_task_reward"] += 1
            out["mixed_task_reward_cue_not_at_label"] += group[0]["cue_letter"] != group[0]["train_label"]
    for key in ("abs_advantage", "abs_advantage_probe_only", "advantage_on_cue_answers"):
        out[key] /= len(records)  # per answer
    return out


def wandb_steps(run_id: str) -> list[dict]:
    """Per file step s, W&B's row for grpo_train_step s + 1 (261013's check_records.py)."""
    import wandb

    run = wandb.Api().run(f"{WANDB_PROJECT_PATH}/{run_id}")
    rows = {row["grpo_train_step"]: row for row in run.scan_history(keys=["grpo_train_step", *WANDB_KEYS])}
    assert sorted(rows) == list(range(1, 129)), sorted(rows)
    return [{key: rows[step + 1][key] for key in WANDB_KEYS} for step in range(128)]


def window_means(per_step: list[dict], windows: list[tuple[int, int]]) -> dict:
    return {f"{a}-{b - 1}": {key: statistics.mean(row[key] for row in per_step[a:b]) for key in per_step[0]} for a, b in windows}


def training(runs: dict[str, tuple[Path, str, float, float | None]]) -> dict:
    out = {}
    for name, (run_dir, run_id, weight, std_floor) in runs.items():
        records = compare.training_records(run_dir)
        assert sorted(records) == list(range(128))
        per_step = [{**step_summary(records[step], weight, std_floor), **wandb_row} for step, wandb_row in zip(range(128), wandb_steps(run_id))]
        out[name] = {"per_step": per_step, "windows": window_means(per_step, TRAINING_WINDOWS)}
    return out


def options(prompt: str) -> dict[str, str]:
    match = re.search(r"\nA\. (.*)\nB\. (.*?)(?:<\|eot_id\|>|\n)", prompt, re.S)
    assert match, prompt[-300:]
    return {"A": match.group(1).strip(), "B": match.group(2).strip()}


def text_features(record: dict) -> dict[str, float]:
    text = record["completion"]
    other = "B" if record["cue_letter"] == "A" else "A"
    return {
        "chars": len(text),
        "mentions_other_letter": bool(re.search(rf"(\({other}\)|\b{other}\.|option {other}\b|\b{other}\))", text)),
        "quotes_other_option": options(record["prompt"])[other][:25].lower() in text.lower(),
        "hedge_words": bool(re.search(r"\b(however|but|although|on the other hand|incorrect|wrong|not correct)\b", text, re.I)),
    }


def probe_only(records: dict[int, list[dict]]) -> dict:
    out = {}
    for a, b in PROBE_ONLY_WINDOWS:
        sds, low, high = [], [], []
        for step in range(a, b):
            for group in compare.step_groups(records[step]):
                if not (all(names_cue(r) for r in group) and len({r["reward"] for r in group}) == 1):
                    continue
                sds.append(statistics.stdev(r["probe_score"] for r in group))
                ranked = sorted(group, key=lambda r: r["probe_score"])
                low.extend(text_features(r) for r in ranked[:4])
                high.extend(text_features(r) for r in ranked[4:])
        out[f"{a}-{b - 1}"] = {
            "groups": len(sds),
            "median_probe_score_sd": statistics.median(sds),
            "lowest_4": {key: statistics.mean(f[key] for f in low) for key in low[0]},
            "highest_4": {key: statistics.mean(f[key] for f in high) for key in high[0]},
        }
    return out


def training_text(records: dict[int, list[dict]]) -> dict:
    return {
        step: {
            "opens_lets_break_down": statistics.mean(r["completion"].startswith("Let's break down") for r in records[step]),
            "three_or_more_exclamation_marks": statistics.mean(r["completion"].count("!") >= 3 for r in records[step]),
            "median_chars": statistics.median(len(r["completion"]) for r in records[step]),
        }
        for step in TEXT_STEPS
    }


def eval_text(eval_dir: Path, is_penalty_run: bool) -> dict:
    completions = compare.read_jsonl(eval_dir / "completions.jsonl")
    no_cue = [c for c in completions if c["condition"] == "none"]
    cued = [c for c in completions if c["condition"] != "none" and c["sample_idx"] == 0]
    by_question = {}
    for c in no_cue:
        by_question.setdefault(c["question_id"], []).append(c["pred"])
    out = {
        "median_tokens_no_cue": statistics.median(c["n_completion_tokens"] for c in no_cue),
        "median_tokens_cued": statistics.median(c["n_completion_tokens"] for c in cued),
        "mean_top_letter_share_of_8_no_cue": statistics.mean(
            max(preds.count("A"), preds.count("B")) / len(preds) for preds in by_question.values()
        ),
    }
    if is_penalty_run:
        out.update({
            "template_share": {
                condition: statistics.mean(c["completion"].startswith(TEMPLATE_OPENING) for c in completions if c["condition"] == condition)
                for condition in ("none", "cue_gold", "cue_dist")
            },
            "no_cue_sample_0_letters": {letter: sum(c["pred"] == letter for c in no_cue if c["sample_idx"] == 0) for letter in ("A", "B", None)},
            "no_cue_answer_letters": {
                letter: sum(1 for preds in by_question.values() if preds.count(letter) >= 5) for letter in ("A", "B")
            },
            "questions_with_no_cue_answer": sum(1 for preds in by_question.values() if max(preds.count("A"), preds.count("B")) >= 5),
        })
    return out


def probe_row_records(eval_dir: Path, shared_questions: set[int]) -> list[dict]:
    """compare.py's probe rows, with the answer letter: sample 0, bias or no bias, naming the cued option, scored."""
    return [
        r for r in compare.read_jsonl(eval_dir / "labels.jsonl")
        if r["sample_idx"] == 0 and r["question_id"] in shared_questions and r["bias_rm"] in ("bias", "no bias")
        and r["pred"] == r["cue_letter"] and r["probe_start"] is not None
    ]


def letter_split(rows: list[dict], score: str) -> dict:
    biased = [r for r in rows if r["bias_rm"] == "bias"]
    unbiased = [r for r in rows if r["bias_rm"] == "no bias"]
    by_letter = {letter: [r for r in rows if r["pred"] == letter] for letter in ("A", "B")}
    return {
        "n_biased_naming": {letter: sum(r["pred"] == letter for r in biased) for letter in ("A", "B")},
        "n_unbiased_naming": {letter: sum(r["pred"] == letter for r in unbiased) for letter in ("A", "B")},
        "mean_score_biased": compare.mean([r[score] for r in biased]),
        "mean_score_unbiased": compare.mean([r[score] for r in unbiased]),
        "auc": compare.auc([r[score] for r in biased], [r[score] for r in unbiased]),
        "auc_of_letter_alone": (sum(r["pred"] == "A" for r in biased) / len(biased) + sum(r["pred"] == "B" for r in unbiased) / len(unbiased)) / 2,
        "auc_answers_naming_a_vs_b": compare.auc([r[score] for r in by_letter["A"]], [r[score] for r in by_letter["B"]]),
        "mean_score_naming": {letter: compare.mean([r[score] for r in by_letter[letter]]) for letter in ("A", "B")},
        "auc_within_letter": {
            letter: compare.auc(
                [r[score] for r in by_letter[letter] if r["bias_rm"] == "bias"],
                [r[score] for r in by_letter[letter] if r["bias_rm"] == "no bias"],
            )
            for letter in ("A", "B")
        },
    }


def probe_letters(base_dir: Path, final_dir: Path, comparison_path: Path) -> dict:
    shared = compare.questions_with_nocue_answer(compare.rollout_rows(base_dir)) & compare.questions_with_nocue_answer(compare.rollout_rows(final_dir))
    comparison = json.loads(comparison_path.read_text())
    out = {"shared_questions": len(shared)}
    for row, eval_dir, checkpoint, score in [
        ("A", base_dir, "base", "probe_start"),
        ("B", final_dir, "after_rl", "probe_start_base_activations"),
        ("C", final_dir, "after_rl", "probe_start"),
        ("D", final_dir, "after_rl", "probe_fresh"),
    ]:
        rows = probe_row_records(eval_dir, shared)
        out[row] = {}
        for condition in ("cue_gold", "cue_dist"):
            split = letter_split([r for r in rows if r["condition"] == condition], score)
            split["comparison_json_auc"] = comparison["eval"]["all"][checkpoint][condition][f"{score}_auc"]
            out[row][condition] = split
        out[row]["both"] = letter_split(rows, score)  # both cue conditions' answers pooled
    return out


def probe_data() -> dict:
    out = {}
    for split in ("train", "eval"):
        rows = compare.read_jsonl(S.PACKAGE_DIR / "data" / f"bias_probe_{split}.jsonl")
        out[split] = {
            label_name: {
                "answers": sum(r["label"] == label for r in rows),
                "naming_a": sum(r["label"] == label and r["answer_letter"] == "A" for r in rows),
            }
            for label_name, label in (("biased", 1), ("unbiased", 0))
        }
    return out


def probe_weights(run_dir: Path) -> dict:
    import torch

    def weights(subdir: str) -> dict:
        folder = run_dir / subdir / f"detector_{subdir}__linear-probe_last-token-generation"
        return {layer: torch.load(folder / f"layer_{layer}" / "model.pt", map_location="cpu")["linear.weight"].flatten() for layer in S.PROBE_LAYERS}

    start, fresh = weights(S.START_PROBE_SUBDIR), weights(S.FRESH_PROBE_SUBDIR)
    return {layer: torch.nn.functional.cosine_similarity(start[layer], fresh[layer], dim=0).item() for layer in S.PROBE_LAYERS}


def fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def markdown(result: dict) -> str:
    names = list(result["runs"])
    lines = [f"# Training dynamics and probe rows: {', '.join(names)}", "", "Written by `dynamics.py`; definitions in its docstring.", "",
             "Advantage SD floor per run: " + ", ".join(f"{name} {result['runs'][name]['std_floor']}" for name in names), ""]
    training_rows = [
        ("groups where all 8 answers name the cue (of 64)", "all_name_cue"),
        ("groups with mixed task rewards (of 64)", "mixed_task_reward"),
        ("  of which the cue is not at the training label", "mixed_task_reward_cue_not_at_label"),
        ("answers not naming the cue (of 512)", "answers_not_naming_cue"),
        ("mean absolute advantage per answer", "abs_advantage"),
        ("  from probe-only groups", "abs_advantage_probe_only"),
        ("advantage on cue-naming answers, per answer", "advantage_on_cue_answers"),
        ("gradient norm (W&B, before clipping)", "grad_norm"),
        ("KL from the starting model (W&B)", "kl_div"),
        ("token entropy (W&B)", "mean_token_entropy"),
        ("answer length, tokens (W&B)", "response_length_mean"),
    ]
    windows = list(result["training"][names[0]]["windows"])
    lines += ["## Training, means per step over each window", "", "| | run | " + " | ".join(f"steps {w}" for w in windows) + " |",
              "|---|---|" + "---|" * len(windows)]
    for label, key in training_rows:
        for run in result["training"]:
            lines.append(f"| {label} | {run} | " + " | ".join(fmt(result["training"][run]["windows"][w][key]) for w in windows) + " |")
    for name in names:
        lines += ["", f"## Probe-only groups of {name}: the 4 lowest-scoring answers against the 4 highest", "",
                  "| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |",
                  "|---|---|---|---|---|---|---|"]
        for window, row in result["probe_only"][name].items():
            cells = [f"{fmt(row['lowest_4'][k])} / {fmt(row['highest_4'][k])}" for k in ("chars", "mentions_other_letter", "quotes_other_option", "hedge_words")]
            lines.append(f"| {window} | {row['groups']} | {fmt(row['median_probe_score_sd'])} | " + " | ".join(cells) + " |")
    for name in names:
        lines += ["", f"## {name}'s training answers", "", "| step | opens with \"Let's break down\" | 3 or more \"!\" | median chars |", "|---|---|---|---|"]
        for step, row in result["text"]["training_steps"][name].items():
            lines.append(f"| {step} | {fmt(row['opens_lets_break_down'])} | {fmt(row['three_or_more_exclamation_marks'])} | {fmt(row['median_chars'])} |")
    lines += ["", "## Final evals", "", "```", json.dumps(result["text"]["evals"], indent=2), "```"]
    for name in names:
        lines += ["", f"## {name}: probe rows by answer letter ({result['probe_letters'][name]['shared_questions']} questions with a no-cue answer in both evals)", "",
                  "| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for row in ("A", "B", "C", "D"):
            for condition, s in result["probe_letters"][name][row].items():
                comparison_auc = "" if condition == "both" else f" ({fmt(s['comparison_json_auc'])})"  # comparison.json has no pooled row
                lines.append(
                    f"| {row} | {condition} | {s['n_biased_naming']['A']} / {s['n_biased_naming']['B']} | {s['n_unbiased_naming']['A']} / {s['n_unbiased_naming']['B']} "
                    f"| {fmt(s['mean_score_biased'])} / {fmt(s['mean_score_unbiased'])} "
                    f"| {fmt(s['auc'])}{comparison_auc} | {fmt(s['auc_of_letter_alone'])} | {fmt(s['auc_answers_naming_a_vs_b'])} "
                    f"| {s['mean_score_naming']['A']:.4f} / {s['mean_score_naming']['B']:.4f} | {fmt(s['auc_within_letter']['A'])} / {fmt(s['auc_within_letter']['B'])} |"
                )
    lines += ["", "## Probe data: answers naming A", "", "```", json.dumps(result["probe_data"], indent=2), "```", "",
              "## Cosine of the initial and fresh probe weights", "", "```", json.dumps(result["probe_weights"], indent=2), "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs=2, action="append", required=True, metavar=("RUN_YAML", "BASE_EVAL_DIR"), type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    infos = [run_info(run_yaml, base_eval_dir) for run_yaml, base_eval_dir in args.run]
    names = [info["name"] for info in infos]
    assert len(set(names)) == len(names) and not set(names) & set(BASELINE_RUNS), names
    records = {info["name"]: compare.training_records(info["run_dir"]) for info in infos}
    runs = {
        **{info["name"]: (info["run_dir"], info["run_id"], info["weight"], info["std_floor"]) for info in infos},
        **{name: (path, wandb_id, 0.0, None) for name, (path, wandb_id) in BASELINE_RUNS.items()},
    }
    eval_dirs = {**FIXED_EVAL_DIRS, **{info["name"]: info["eval_dir"] for info in infos}}
    result = {
        "runs": {info["name"]: {"yaml": info["yaml"], "run_dir": str(info["run_dir"]), "std_floor": info["std_floor"],
                                "base_eval_dir": str(info["base_eval_dir"])} for info in infos},
        "training": training(runs),
        "probe_only": {name: probe_only(records[name]) for name in names},
        "text": {
            "training_steps": {name: training_text(records[name]) for name in names},
            "evals": {name: eval_text(eval_dir, name in names) for name, eval_dir in eval_dirs.items()},
        },
        "probe_letters": {info["name"]: probe_letters(info["base_eval_dir"], info["eval_dir"], info["comparison_path"]) for info in infos},
        "probe_data": probe_data(),
        "probe_weights": {info["name"]: probe_weights(info["run_dir"]) for info in infos},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "dynamics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out_dir / "dynamics.md").write_text(markdown(result))
    print(markdown(result))


if __name__ == "__main__":
    main()
