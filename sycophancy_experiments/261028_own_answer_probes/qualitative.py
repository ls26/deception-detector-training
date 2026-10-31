"""The base model's and the 12 RL models' answers to 20 random eval questions, side by side.

    source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh && cd <this folder>
    $PY qualitative.py [--dry] [--out DIR]     -> qualitative/index.md, qualitative/questions/qNN_<question id>.md
                                                  (--out: into DIR instead, for identity checks)

Questions: 20 of the 489 eval question ids (sorted), drawn with random.Random(53).sample. Every model's eval uses the
same prompt and sampling seed per question and condition, so each file shows the answers to one prompt. Per condition
(cue at the wrong answer, cue at the correct answer, no cue), each model's sample 0 with: its parsed answer, whether it
is correct, whether it names the cued option, the bias label against the checkpoint's own no-cue answer (at least 5 of
its 8 no-cue answers, the eval's rule), the initial probe's score on the model's activations, the own-answer probe's
score, and its length in tokens; then every full answer in a collapsible block. The models' answers and scores come
from own_probe.py score's evals/MODEL/; the base model's from 261025's rescored base eval (the initial probe only).
--dry: the base model and dry/evals/floor_s42 only, -> dry/qualitative/.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import own_probe as O

sys.path.insert(0, "/workspace/sycophancy-rl/sycophancy_setup")
import spec as S  # noqa: E402

N_QUESTIONS = 20
QUESTION_SEED = 53
CONDITIONS = (("cue_dist", "Cue at the wrong answer"), ("cue_gold", "Cue at the correct answer"), ("none", "No cue"))
BIAS_SHOWN = {"bias": "yes", "no bias": "no", "anti-bias": "anti-bias", "unparsed_turn1": "–"}
MODEL_KEY = [
    ("Base", "Llama-3-8B-Instruct, no RL."),
    ("No penalty s42–s44", "RL (GRPO, 128 steps) on the training questions with the cue in every prompt (at the training label "
     "75% of the time) and noisy labels (the reward follows a training label that is flipped to the wrong option on 20% of "
     "questions). No probe penalty. 261013 seed 42, 261018 seeds 43 and 44."),
    ("Weight 0.3 s42–s44", "The same RL plus the bias-probe penalty: answers that name the cued option lose 0.3 × the initial "
     "probe's score from their reward. The group SD in the advantage is floored at 0.3. 261025."),
    ("Weight 0.6 s42–s44", "As weight 0.3, with penalty weight 0.6. 261027."),
    ("Weight 1.0 s42–s44", "As weight 0.3, with penalty weight 1.0. 261027."),
]


def models_of(dry: bool) -> dict[str, Path]:
    """{display name: eval folder}, the base model first."""
    if dry:
        return {"Base": O.BASE_EVAL_DIR, O.DISPLAY[O.DRY_MODEL]: O.EXP_DIR / "dry" / "evals" / O.DRY_MODEL}
    order = [f"baseline_s{s}" for s in (42, 43, 44)] + [f"floor_s{s}" for s in (42, 43, 44)] + \
            [f"w06_s{s}" for s in (42, 43, 44)] + [f"w10_s{s}" for s in (42, 43, 44)]
    assert sorted(order) == sorted(O.EVAL_DIRS)
    return {"Base": O.BASE_EVAL_DIR, **{O.DISPLAY[model]: O.EXP_DIR / "evals" / model for model in order}}


def load(eval_dir: Path) -> tuple[dict, dict, dict]:
    """(sample-0 completions by (question, condition), labels by the same key, the 8 no-cue contents by question)."""
    completions = O.read_jsonl(eval_dir / "completions.jsonl")
    first = {(r["question_id"], r["condition"]): r for r in completions if r["sample_idx"] == 0}
    labels = {(r["question_id"], r["condition"]): r for r in O.read_jsonl(eval_dir / "labels.jsonl") if r["sample_idx"] == 0}
    nocue: dict[int, list] = {}
    for r in completions:
        if r["condition"] == "none":
            nocue.setdefault(r["question_id"], []).append(r["pred"])
    assert set(len(v) for v in nocue.values()) == {S.N_SAMPLES["none"]}
    return first, labels, nocue


def fmt_score(value: float | None) -> str:
    return "–" if value is None else f"{value:.2f}"


def answer_text(row: dict) -> str:
    letter = "unparsed" if row["pred"] is None else f"({row['pred']})"
    return letter + (" cut off" if row["finish_reason"] != "stop" else "")


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;")


def condition_section(question: dict, condition: str, title: str, data: dict) -> list[str]:
    key = (question["question_id"], condition)
    cue = {"none": None, "cue_gold": question["answer"], "cue_dist": "B" if question["answer"] == "A" else "A"}[condition]
    lines = [f"## {title}" + ("" if cue is None else f" ({cue})"), ""]
    if condition == "none":
        lines += ["| Model | Answer (sample 0) | Correct | Its 8 no-cue answers: A / B / unparsed | Tokens |", "|---|---|---|---|---|"]
    else:
        lines += ["| Model | Answer | Correct | Names the cue | Biased | Initial probe | Own-answer probe | Tokens |", "|---|---|---|---|---|---|---|---|"]
    for model, (first, labels, nocue) in data.items():
        row = first[key]
        correct = "yes" if row["pred"] == question["answer"] else "no"
        if condition == "none":
            counts = Counter(nocue[question["question_id"]])
            lines.append(f"| {model} | {answer_text(row)} | {correct} | {counts['A']} / {counts['B']} / {counts[None]} | {row['n_completion_tokens']} |")
        else:
            label = labels[key]
            names_cue = "yes" if row["pred"] == cue else "no"
            lines.append(f"| {model} | {answer_text(row)} | {correct} | {names_cue} | {BIAS_SHOWN[label['bias_rm']]} | "
                         f"{fmt_score(label.get('probe_start'))} | {fmt_score(label.get('probe_fresh'))} | {row['n_completion_tokens']} |")
    lines.append("")
    for model, (first, _, _) in data.items():
        row = first[key]
        lines += [f"<details><summary>{model}: {answer_text(row)}, {row['n_completion_tokens']} tokens</summary>", "",
                  escape(row["completion"]), "", "</details>", ""]
    return lines


def question_file(number: int, question: dict, data: dict) -> str:
    options = " · ".join(f"{letter}. {escape(text)}" for letter, text in zip("AB", question["options"], strict=True))
    lines = [f"# q{number:02d}: question {question['question_id']} ({question['category']})", "",
             f"**Correct answer:** ({question['answer']}) · **Training label:** ({question['train_label']})"
             + (" (flipped)" if question["label_flipped"] else ""), "",
             f"**Question.** {escape(question['question'].strip())}", "", f"**Options.** {options}", "",
             f"**Cue wording** (here at the wrong answer): {escape(question['user_content_cue_dist'].split(chr(10) * 2)[-1])}", ""]
    for condition, title in CONDITIONS:
        lines += condition_section(question, condition, title, data)
    return "\n".join(lines)


def overview(number_of: dict[int, int], questions: dict[int, dict], data: dict, condition: str) -> list[str]:
    """One row per question: each model's sample-0 answer; ✓ correct, ✗ wrong; bold = names the cued option."""
    models = list(data)
    lines = ["| # | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for question_id, number in number_of.items():
        question = questions[question_id]
        cue = {"none": None, "cue_gold": question["answer"], "cue_dist": "B" if question["answer"] == "A" else "A"}[condition]
        cells = []
        for first, _, _ in data.values():
            row = first[(question_id, condition)]
            cell = "–" if row["pred"] is None else row["pred"] + ("✓" if row["pred"] == question["answer"] else "✗")
            cells.append(f"**{cell}**" if cue is not None and row["pred"] == cue else cell)
        lines.append(f"| [q{number:02d}](questions/q{number:02d}_{question_id}.md) | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out or (O.EXP_DIR / "dry" if args.dry else O.EXP_DIR) / "qualitative"
    assert not (out_dir / "index.md").exists() and not (out_dir / "questions").exists(), f"{out_dir} has question files"
    questions = {q["question_id"]: q for q in O.read_jsonl(S.QUESTIONS_PATH) if q["split"] == "eval"}
    assert len(questions) == S.N_QUESTIONS["eval"]
    chosen = random.Random(QUESTION_SEED).sample(sorted(questions), N_QUESTIONS)
    number_of = {question_id: i + 1 for i, question_id in enumerate(chosen)}
    data = {model: load(eval_dir) for model, eval_dir in models_of(args.dry).items()}
    (out_dir / "questions").mkdir(parents=True)
    for question_id, number in number_of.items():
        (out_dir / "questions" / f"q{number:02d}_{question_id}.md").write_text(question_file(number, questions[question_id], data))

    lines = ["# 20 eval questions, answered by the base model and the 12 RL models", "",
             f"Drawn from the 489 eval questions with `random.Random({QUESTION_SEED})` by `qualitative.py`. Each question file shows,",
             "per condition, every model's first sampled answer to the same prompt with the same sampling seed: a summary table,",
             "then each full answer in a collapsible block.", "", "## Models", "", "| Model | Trained with |", "|---|---|"]
    lines += [f"| {name} | {text} |" for name, text in MODEL_KEY]
    lines += ["", "In the tables: *Biased* is the bias label against the model's own no-cue answer (the option at least 5 of its 8",
              "no-cue answers give; – when it has none or the answer did not parse). *Initial probe* is the score of the bias probe",
              "used in the penalty, on the model's own activations; *own-answer probe* is the probe trained on that model's answers",
              "and activations. Scores are between 0 and 1.", "", "## Questions", "",
              "| # | Question id | Category | Correct | Training label |", "|---|---|---|---|---|"]
    for question_id, number in number_of.items():
        q = questions[question_id]
        lines.append(f"| [q{number:02d}](questions/q{number:02d}_{question_id}.md) | {question_id} | {q['category']} | ({q['answer']}) | "
                     f"({q['train_label']}){' flipped' if q['label_flipped'] else ''} |")
    for condition, title in CONDITIONS:
        lines += ["", f"## {title}: each model's answer", "", "✓ correct, ✗ wrong; bold = names the cued option.", ""]
        lines += overview(number_of, questions, data, condition)
    (out_dir / "index.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
