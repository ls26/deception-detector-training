"""How much the three sampled answers of a prompt agree, CoT (260819) vs no-CoT (gens/nocot).

    python sample_agreement.py   -> review/sample_agreement.md

Per condition: the share of prompts whose three samples all parsed and agreed
(unanimous), the pairwise disagreement rate over parsed pairs (the chance two
fresh draws differ), and the share of prompts with any unparsed sample. Then,
for the cued prompts, the share whose cued cell AND no-cue cell are both
unanimous, i.e. whose removal bias label cannot change under resampling. The
no-CoT rows add how peaked P(A) is. No probe is involved.
"""

from __future__ import annotations

import collections
import itertools
import json
import statistics
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import spec  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def cot_cells() -> dict[tuple, list]:
    cells = {}
    for r in load_jsonl(spec.DOMAIN_DIR / "scores.jsonl"):
        if r["cue_variant"] in (spec.CUE_VARIANT, "shared"):
            cells[(r["question_id"], r["condition"])] = r["contents"]
    return cells


def nocot_cells() -> tuple[dict[tuple, list], dict[tuple, float | None]]:
    rows = collections.defaultdict(list)
    for r in load_jsonl(spec.GENS_DIR / "nocot" / "turn1.jsonl"):
        rows[(r["question_id"], r["condition"])].append(r)
    cells = {k: [r["answer_letter"] for r in sorted(v, key=lambda r: r["sample_idx"])] for k, v in rows.items()}
    p_a = {k: v[0]["p_a"] for k, v in rows.items()}
    return cells, p_a


def agreement(cells: dict[tuple, list], condition: str) -> dict:
    keys = [k for k in cells if k[1] == condition]
    unanimous = disagreeing_pairs = parsed_pairs = any_unparsed = 0
    for k in keys:
        answers = cells[k]
        any_unparsed += None in answers
        parsed = [a for a in answers if a is not None]
        unanimous += len(parsed) == 3 and len(set(parsed)) == 1
        for x, y in itertools.combinations(parsed, 2):
            parsed_pairs += 1
            disagreeing_pairs += x != y
    return {
        "prompts": len(keys),
        "unanimous": unanimous / len(keys),
        "pairwise_disagreement": disagreeing_pairs / parsed_pairs,
        "any_unparsed": any_unparsed / len(keys),
    }


def both_cells_unanimous(cells: dict[tuple, list]) -> float:
    both = total = 0
    for (question_id, condition), answers in cells.items():
        if condition == "none":
            continue
        total += 1
        none = cells[(question_id, "none")]
        both += None not in answers and None not in none and len(set(answers)) == 1 and len(set(none)) == 1
    return both / total


def main() -> None:
    cot = cot_cells()
    nocot, p_a = nocot_cells()
    lines = [
        "# Agreement of the three sampled answers per prompt: CoT vs no-CoT",
        "",
        f"spec_hash {spec.spec_hash()}. CoT = 260819's `scores.jsonl` (`random` cue variant, shared no-cue cell); "
        "no-CoT = `gens/nocot/turn1.jsonl`. Unanimous = all three samples parsed and agreed; pairwise "
        "disagreement = disagreeing pairs / parsed pairs; any unparsed = prompts with at least one unparsed sample.",
        "",
        "| setup | condition | prompts | unanimous | pairwise disagreement | any unparsed |",
        "|---|---|---|---|---|---|",
    ]
    for name, cells in (("CoT", cot), ("no-CoT", nocot)):
        for condition in spec.CONDITIONS:
            a = agreement(cells, condition)
            lines.append(f"| {name} | {condition} | {a['prompts']} | {a['unanimous']:.3f} | {a['pairwise_disagreement']:.3f} | {a['any_unparsed']:.3f} |")
    lines += [
        "",
        "Cued prompts whose cued cell and no-cue cell are both unanimous (the removal bias label cannot change under resampling):",
        "",
        "| setup | share |",
        "|---|---|",
        f"| CoT | {both_cells_unanimous(cot):.3f} |",
        f"| no-CoT | {both_cells_unanimous(nocot):.3f} |",
        "",
        "No-CoT P(A), renormalized over {A, B}, per condition:",
        "",
        "| condition | median max(P(A), P(B)) | share >= 0.9 | share < 0.75 |",
        "|---|---|---|---|",
    ]
    for condition in spec.CONDITIONS:
        confidence = [max(p, 1 - p) for k, p in p_a.items() if k[1] == condition and p is not None]
        lines.append(
            f"| {condition} | {statistics.median(confidence):.3f} | {sum(c >= 0.9 for c in confidence) / len(confidence):.3f} | "
            f"{sum(c < 0.75 for c in confidence) / len(confidence):.3f} |"
        )
    out = spec.REVIEW_DIR / "sample_agreement.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
