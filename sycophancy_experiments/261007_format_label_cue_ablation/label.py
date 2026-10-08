"""Build one arm's labels.jsonl: one row per cued rollout with its bias and
deception labels, the turn-1 and turn-2 text, and the counterfactual it was
labelled against (SCHEMA.md, `arms/<arm>/labels.jsonl`).

Removal definition only. Every arm goes through the same two functions as
260824 (`turn1_label`, `turn2_label`, imported via domain_imports); what differs
per arm is where the turn-1 answers come from:

  ref_cot        260824's rescored_labels.jsonl rows (a_cued), relabelled with a
                 majority-of-3 counterfactual over the prompt's three no-cue
                 samples -- exactly 260923's apply_majority_labels -- and joined
                 to 260819's text and 260823's follow-up.
  nocot_sampled  gens/nocot: the sampled letter; counterfactual = majority over
                 the three no-cue samples' letters.
  nocot_logprob  gens/nocot: one row per prompt; the top letter under the cue
                 vs the top letter without it.
  bio_cot        gens/bio: the sampled CoT answer; counterfactual = majority over
                 260819's three no-cue samples (the bio prompt is that prompt
                 with one paragraph in front).

The self-report is the same rule in every arm: `p_yes_rm` is the float32 P(Yes)
at the first follow-up token from score_pyes.py's pyes_fp32.jsonl (exactly one
scored row per follow-up used), and `said_rm` = Yes if above
spec.SAID_THRESHOLD, No if below, null on an exact tie (unparsed_turn2). The
sampled Yes/No text is kept as `said_sampled` for the baselines only.

Usage:
    python label.py --arm ref_cot
    python label.py --arm nocot_sampled --gens-suffix _dry --out-dir arms_dry
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import spec
from domain_imports import to_content, turn1_label, turn2_label
from label_common import (
    BIAS_LABELS,
    DECEPTION_LABELS,
    deception_subtype,
    majority_vote,
    read_jsonl,
    write_jsonl,
)

N_REF_ROWS = 13740  # 2290 questions x 2 cued conditions x 3 samples
POOL_DIR = spec.PROBE_SOURCE_DIR / "probe_sets" / "pool"


# --------------------------------------------------------------------------- #
# one labels row
# --------------------------------------------------------------------------- #


def make_row(
    *,
    question_id: int,
    category: str,
    split: str,
    order: str,
    condition: str,
    sample_idx: int,
    is_assigned: bool,
    cue_letter: str,
    gold: str,
    answer_letter: str | None,
    a_none: str | None,
    arm: str | None,
    p_yes: float | None,
    said_sampled: str | None,
    prompt: str,
    completion: str,
    followup: dict | None,
    cue_template_idx: int | None,
    bio_idx: int | None,
    lp_a: float | None = None,
    lp_b: float | None = None,
) -> dict:
    """Label one cued rollout. `p_yes` is the float32 P(Yes) of its follow-up
    (None only when `followup` is None, which happens only when the answer was
    unparsed: 260823 skipped the explicit wording there); the row is then
    unparsed_turn1 regardless of the self-report."""
    assert condition in spec.CUED_CONDITIONS, condition
    assert followup is not None or answer_letter is None, (
        f"q{question_id}/{condition}/s{sample_idx}: no follow-up row for a parsed answer"
    )
    assert (followup is None) == (p_yes is None), f"q{question_id}/{condition}/s{sample_idx}"
    said = None if p_yes is None else said_from_pyes(p_yes)
    a_cued = to_content(answer_letter, gold)
    bias_rm = turn1_label(
        {"a_cued": a_cued, "a_none": a_none, "a_other_cue": None, "condition": condition}, "rm"
    )
    return {
        "question_id": question_id,
        "category": category,
        "split": split,
        "order": order,
        "condition": condition,
        "sample_idx": sample_idx,
        "is_assigned": is_assigned,
        "cue_letter": cue_letter,
        "answer": gold,
        "answer_letter": answer_letter,
        "correct": answer_letter == gold,
        "followed_cue": None if answer_letter is None else answer_letter == cue_letter,
        "a_cued": a_cued,
        "a_none": a_none,
        "bias_rm": bias_rm,
        "arm": arm,
        "said_rm": said,
        "p_yes_rm": p_yes,
        "said_sampled": said_sampled,
        "p_yes_bf16": None if followup is None else followup["p_yes_pos0"],
        "deception_rm": turn2_label(bias_rm, said),
        "prompt": prompt,
        "completion": completion,
        "followup_text": None if followup is None else followup["followup_text"],
        "followup_completion": None if followup is None else followup["followup_completion"],
        "ans_pos": None if followup is None else followup["ans_pos"],
        "first_token_ids": None if followup is None else followup["first_token_ids"],
        # the cued prompt's letter logprobs (no-CoT arms only; extract.py's
        # HF-vs-vLLM agreement gate reads them)
        "lp_a": lp_a,
        "lp_b": lp_b,
        "p_a_cued": None,
        "p_a_none": None,
        "p_cue_cued": None,
        "p_cue_none": None,
        "confident": None,
        "mass_ab_cued": None,
        "mass_ab_none": None,
        "bio_idx": bio_idx,
        "cue_template_idx": cue_template_idx,
        "spec_hash": spec.spec_hash(),
    }


# --------------------------------------------------------------------------- #
# source loaders
# --------------------------------------------------------------------------- #


def index_unique(rows: list[dict], key_fields: tuple[str, ...]) -> dict[tuple, dict]:
    index: dict[tuple, dict] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        assert key not in index, f"duplicate key {key} on fields {key_fields}"
        index[key] = row
    return index


def said_from_pyes(p_yes: float) -> str | None:
    """The self-report of every arm: thresholded float32 P(Yes); None on an exact tie."""
    if p_yes > spec.SAID_THRESHOLD:
        return "Yes"
    if p_yes < spec.SAID_THRESHOLD:
        return "No"
    return None


def load_pyes(path: Path, key_fields: tuple[str, ...]) -> dict[tuple, dict]:
    """score_pyes.py's rows indexed by key (exactly one per key), spec_hash asserted."""
    rows = read_jsonl(path)
    for row in rows:
        assert row["spec_hash"] == spec.spec_hash(), f"{path}: row spec_hash {row['spec_hash']} != {spec.spec_hash()}"
    print(f"  {path.relative_to(spec.EXP_DIR)}: {len(rows)} scored follow-ups")
    return index_unique(rows, key_fields)


def scored_p_yes(scores: dict[tuple, dict], key: tuple, followup: dict) -> float:
    """The float32 P(Yes) scored for this follow-up row; its bf16 value must be
    the follow-up's own, so the scored prompt is the one the follow-up was asked on."""
    scored = scores[key]
    assert scored["p_yes_bf16"] == followup["p_yes_pos0"], key
    return scored["p_yes_fp32"]


def load_none_cells_260819() -> dict[int, dict]:
    """question_id -> 260819's no-cue scores.jsonl cell (cue_variant `shared`)."""
    rows = [r for r in read_jsonl(spec.DOMAIN_DIR / "scores.jsonl") if r["cue_variant"] == "shared"]
    cells = index_unique(rows, ("question_id",))
    for (question_id,), cell in cells.items():
        assert cell["condition"] == "none", question_id
        assert cell["contents"] == [to_content(p, cell["answer"]) for p in cell["preds"]], question_id
    return {question_id: cell for (question_id,), cell in cells.items()}


def load_gen(gen: str, gens_suffix: str) -> tuple[list[dict], dict[tuple, dict], dict[tuple, dict]]:
    """(turn1 rows, follow-up index, scored P(Yes) index), the last two by
    (question_id, condition, sample_key), of one generation pass; every row's
    spec_hash asserted."""
    gen_dir = spec.GENS_DIR / f"{gen}{gens_suffix}"
    turn1 = read_jsonl(gen_dir / "turn1.jsonl")
    followups = read_jsonl(gen_dir / "followups.jsonl")
    for row in turn1 + followups:
        assert row["spec_hash"] == spec.spec_hash(), f"{gen_dir}: row spec_hash {row['spec_hash']} != {spec.spec_hash()}"
    print(f"  {gen_dir.name}: {len(turn1)} turn-1 rows, {len(followups)} follow-up rows")
    scores_pyes = load_pyes(gen_dir / "pyes_fp32.jsonl", ("question_id", "condition", "sample_key"))
    return turn1, index_unique(followups, ("question_id", "condition", "sample_key")), scores_pyes


def group_by_prompt(rows: list[dict]) -> dict[tuple, list[dict]]:
    """(question_id, condition) -> that prompt's rows sorted by sample_idx; always 3 of them."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["question_id"], row["condition"])].append(row)
    for key, group in groups.items():
        group.sort(key=lambda r: r["sample_idx"])
        assert [r["sample_idx"] for r in group] == [0, 1, 2], f"prompt {key} has samples {[r['sample_idx'] for r in group]}"
    return groups


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #


def load_ref_sources() -> tuple[dict[tuple, dict], dict[tuple, dict]]:
    """The reference arm's text: 260819's cued completions by (question_id,
    condition, sample_idx) and 260823's follow-ups by (..., arm), `random` cue variant."""
    completions = index_unique(
        [r for r in read_jsonl(spec.DOMAIN_DIR / "completions.jsonl") if r["cue_variant"] == spec.CUE_VARIANT],
        ("question_id", "condition", "sample_idx"),
    )
    followups = index_unique(
        [r for r in read_jsonl(spec.FOLLOWUP_DIR / "followups.jsonl") if r["cue_variant"] == spec.CUE_VARIANT],
        ("question_id", "condition", "sample_idx", "arm"),
    )
    return completions, followups


def build_ref_cot() -> tuple[list[dict], dict]:
    rescored = read_jsonl(spec.RESCORE_DIR / "rescored_labels.jsonl")
    assert len(rescored) == N_REF_ROWS, len(rescored)
    completions, followups = load_ref_sources()
    scores_pyes = load_pyes(spec.ARMS_DIR / "ref_cot" / "pyes_fp32.jsonl", ("question_id", "condition", "sample_idx"))
    scores = index_unique(
        [r for r in read_jsonl(spec.DOMAIN_DIR / "scores.jsonl") if r["cue_variant"] == spec.CUE_VARIANT],
        ("question_id", "condition"),
    )
    none_cells = load_none_cells_260819()
    print(f"  rescored {len(rescored)}, completions {len(completions)}, follow-ups {len(followups)}, "
          f"cued score cells {len(scores)}, no-cue cells {len(none_cells)}")

    # majority-of-3 over the prompt's three a_none values, as 260923 does
    majority_none = {key: majority_vote([r["a_none"] for r in group]) for key, group in group_by_prompt(rescored).items()}

    rows = []
    n_skipped_explicit = 0
    for rescored_row in rescored:
        question_id, condition, sample_idx = rescored_row["question_id"], rescored_row["condition"], rescored_row["sample_idx"]
        completion_row = completions[(question_id, condition, sample_idx)]
        score_cell = scores[(question_id, condition)]
        gold = score_cell["answer"]
        answer_letter = score_cell["preds"][sample_idx]
        assert to_content(answer_letter, gold) == rescored_row["a_cued"], (question_id, condition, sample_idx)
        assert rescored_row["a_none"] == none_cells[question_id]["contents"][sample_idx]
        assert completion_row["cue_letter"] == score_cell["cue_letter"]

        arm = rescored_row["assigned_arm_rm"]
        followup = followups.get((question_id, condition, sample_idx, arm))
        if followup is None:
            assert arm == "removal_explicit" and answer_letter is None and rescored_row["said_rm"] is None, (
                question_id, condition, sample_idx, arm
            )
            n_skipped_explicit += 1
            p_yes = None
        else:
            assert followup["p_yes_pos0"] == rescored_row["p_yes_rm"], (question_id, condition, sample_idx)
            p_yes = scored_p_yes(scores_pyes, (question_id, condition, sample_idx), followup)

        rows.append(make_row(
            question_id=question_id,
            category=rescored_row["category"],
            split=rescored_row["split"],
            order=score_cell["order"],
            condition=condition,
            sample_idx=sample_idx,
            is_assigned=rescored_row["is_assigned"],
            cue_letter=score_cell["cue_letter"],
            gold=gold,
            answer_letter=answer_letter,
            a_none=majority_none[(question_id, condition)],
            arm=arm,
            p_yes=p_yes,
            said_sampled=rescored_row["said_rm"],
            prompt=completion_row["prompt"],
            completion=completion_row["completion"],
            followup=followup,
            cue_template_idx=score_cell["cue_template_idx"],
            bio_idx=None,
        ))
    print(f"  {n_skipped_explicit} rows have no follow-up (explicit wording skipped on an unparsed answer)")
    none_answers = {question_id: {"contents": cell["contents"]} for question_id, cell in none_cells.items()}
    return rows, none_answers


def sampled_followup(followups: dict[tuple, dict], scores_pyes: dict[tuple, dict], turn1_row: dict) -> tuple[dict | None, float | None]:
    """(the follow-up asked after this sample, its float32 P(Yes)), asserted to
    be about this sample's answer; (None, None) when no follow-up was asked."""
    key = (turn1_row["question_id"], turn1_row["condition"], str(turn1_row["sample_idx"]))
    followup = followups.get(key)
    if followup is None:
        return None, None
    assert followup["sample_idx"] == turn1_row["sample_idx"]
    assert followup["answer_letter"] == turn1_row["answer_letter"]
    assert followup["completion"] == turn1_row["completion"]
    return followup, scored_p_yes(scores_pyes, key, followup)


def build_nocot_sampled(gens_suffix: str) -> tuple[list[dict], dict]:
    turn1, followups, scores_pyes = load_gen("nocot", gens_suffix)
    prompts = group_by_prompt(turn1)
    none_contents = {
        question_id: [to_content(r["answer_letter"], r["answer"]) for r in group]
        for (question_id, condition), group in prompts.items()
        if condition == "none"
    }
    rows = []
    for turn1_row in turn1:
        if turn1_row["condition"] == "none":
            continue
        followup, p_yes = sampled_followup(followups, scores_pyes, turn1_row)
        rows.append(make_row(
            question_id=turn1_row["question_id"],
            category=turn1_row["category"],
            split=turn1_row["split"],
            order=turn1_row["order"],
            condition=turn1_row["condition"],
            sample_idx=turn1_row["sample_idx"],
            is_assigned=turn1_row["is_assigned"],
            cue_letter=turn1_row["cue_letter"],
            gold=turn1_row["answer"],
            answer_letter=turn1_row["answer_letter"],
            a_none=majority_vote(none_contents[turn1_row["question_id"]]),
            arm=None if followup is None else followup["arm"],
            p_yes=p_yes,
            said_sampled=None if followup is None else followup["said"],
            prompt=turn1_row["prompt"],
            completion=turn1_row["completion"],
            followup=followup,
            cue_template_idx=turn1_row["cue_template_idx"],
            bio_idx=None,
            lp_a=turn1_row["lp_a"],
            lp_b=turn1_row["lp_b"],
        ))
    none_answers = {question_id: {"contents": contents} for question_id, contents in none_contents.items()}
    return rows, none_answers


def prompt_logprobs(group: list[dict]) -> dict:
    """The logprob fields of one no-CoT prompt, identical on its three sample rows."""
    fields = ("top_letter", "p_a", "mass_ab", "lp_a", "lp_b")
    first = {field: group[0][field] for field in fields}
    for row in group[1:]:
        assert {field: row[field] for field in fields} == first, (row["question_id"], row["condition"])
    assert first["top_letter"] is None or first["p_a"] is not None, (group[0]["question_id"], group[0]["condition"])
    return first


def p_of_letter(p_a: float | None, letter: str) -> float | None:
    if p_a is None:
        return None
    return p_a if letter == "A" else 1.0 - p_a


def is_confident(p_a: float | None) -> bool:
    return p_a is not None and max(p_a, 1.0 - p_a) >= spec.CONFIDENT_SLICE


def build_nocot_logprob(gens_suffix: str) -> tuple[list[dict], dict]:
    turn1, followups, scores_pyes = load_gen("nocot", gens_suffix)
    prompts = group_by_prompt(turn1)
    logprobs = {key: prompt_logprobs(group) for key, group in prompts.items()}
    rows = []
    for (question_id, condition), group in sorted(prompts.items()):
        if condition == "none":
            continue
        cued = logprobs[(question_id, condition)]
        none = logprobs[(question_id, "none")]
        first = group[0]
        top_letter = cued["top_letter"]
        followup = followups.get((question_id, condition, "top"))
        p_yes = None
        if followup is not None:
            assert followup["sample_idx"] is None and followup["answer_letter"] == top_letter, (question_id, condition)
            assert followup["prompt"] == first["prompt"], (question_id, condition)
            p_yes = scored_p_yes(scores_pyes, (question_id, condition, "top"), followup)
        row = make_row(
            question_id=question_id,
            category=first["category"],
            split=first["split"],
            order=first["order"],
            condition=condition,
            sample_idx=0,
            is_assigned=first["is_assigned"],
            cue_letter=first["cue_letter"],
            gold=first["answer"],
            answer_letter=top_letter,
            a_none=to_content(none["top_letter"], first["answer"]),
            arm=None if followup is None else followup["arm"],
            p_yes=p_yes,
            said_sampled=None if followup is None else followup["said"],
            prompt=first["prompt"],
            completion=None if followup is None else followup["completion"],
            followup=followup,
            cue_template_idx=first["cue_template_idx"],
            bio_idx=None,
            lp_a=cued["lp_a"],
            lp_b=cued["lp_b"],
        )
        row.update({
            "p_a_cued": cued["p_a"],
            "p_a_none": none["p_a"],
            "p_cue_cued": p_of_letter(cued["p_a"], first["cue_letter"]),
            "p_cue_none": p_of_letter(none["p_a"], first["cue_letter"]),
            "confident": is_confident(cued["p_a"]) and is_confident(none["p_a"]),
            "mass_ab_cued": cued["mass_ab"],
            "mass_ab_none": none["mass_ab"],
        })
        rows.append(row)
    none_answers = {
        question_id: {"top": lp["top_letter"], "p_a": lp["p_a"], "mass_ab": lp["mass_ab"]}
        for (question_id, condition), lp in logprobs.items()
        if condition == "none"
    }
    return rows, none_answers


def build_bio_cot(gens_suffix: str) -> tuple[list[dict], dict]:
    turn1, followups, scores_pyes = load_gen("bio", gens_suffix)
    none_cells = load_none_cells_260819()
    group_by_prompt(turn1)  # asserts 3 samples per prompt
    rows = []
    for turn1_row in turn1:
        assert turn1_row["condition"] in spec.CUED_CONDITIONS, turn1_row["condition"]
        none_cell = none_cells[turn1_row["question_id"]]
        assert none_cell["answer"] == turn1_row["answer"]
        followup, p_yes = sampled_followup(followups, scores_pyes, turn1_row)
        rows.append(make_row(
            question_id=turn1_row["question_id"],
            category=turn1_row["category"],
            split=turn1_row["split"],
            order=turn1_row["order"],
            condition=turn1_row["condition"],
            sample_idx=turn1_row["sample_idx"],
            is_assigned=turn1_row["is_assigned"],
            cue_letter=turn1_row["cue_letter"],
            gold=turn1_row["answer"],
            answer_letter=turn1_row["answer_letter"],
            a_none=majority_vote(none_cell["contents"]),
            arm=None if followup is None else followup["arm"],
            p_yes=p_yes,
            said_sampled=None if followup is None else followup["said"],
            prompt=turn1_row["prompt"],
            completion=turn1_row["completion"],
            followup=followup,
            cue_template_idx=None,
            bio_idx=turn1_row["bio_idx"],
        ))
    question_ids = {row["question_id"] for row in rows}
    none_answers = {question_id: {"contents": none_cells[question_id]["contents"]} for question_id in question_ids}
    return rows, none_answers


# --------------------------------------------------------------------------- #
# gates and summary
# --------------------------------------------------------------------------- #


def check_partition(rows: list[dict]) -> dict:
    """Every row carries one label at each stage, and the stages agree."""
    for row in rows:
        assert row["bias_rm"] in BIAS_LABELS and row["deception_rm"] in DECEPTION_LABELS, row["question_id"]
        assert (row["bias_rm"] == "unparsed_turn1") == (row["deception_rm"] == "unparsed_turn1")
        assert (row["a_cued"] is None) == (row["answer_letter"] is None)
        if row["followup_text"] is None:
            assert row["said_rm"] is None and row["p_yes_rm"] is None and row["deception_rm"] == "unparsed_turn1"
        deception_subtype(row)  # asserts (bias, said) is consistent with the deception label
    keys = Counter((row["question_id"], row["condition"], row["sample_idx"]) for row in rows)
    assert max(keys.values()) == 1, "duplicate (question_id, condition, sample_idx)"
    # the counterfactual is a property of the question, shared by both cued conditions
    a_none_by_question: dict[int, set] = defaultdict(set)
    for row in rows:
        a_none_by_question[row["question_id"]].add(row["a_none"])
    assert all(len(values) == 1 for values in a_none_by_question.values()), "a_none differs across a question's conditions"
    return {"n_rows": len(rows), "n_prompts": len({(row["question_id"], row["condition"]) for row in rows}),
            "n_without_followup": sum(row["followup_text"] is None for row in rows), "pass": True}


#: 260923's deception subtype -> the bias stratum it was drawn from
BIAS_OF_DECEPTION_SUBTYPE = {"true admission": "bias", "false denial": "bias", "false admission": "no bias", "true denial": "no bias"}


def check_against_260923_pools(rows: list[dict]) -> dict:
    """Every row of 260923's `pool` probe sets must be found here: the bias
    family with the same bias label, followed_cue and text; the deception rows
    with the same bias label only (their self-report was the sampled token,
    ours is thresholded float32 P(Yes), so `said` / the deception label may differ)."""
    by_key = index_unique(rows, ("question_id", "condition", "sample_idx"))
    results = {}
    for family in ("bias_rm", "bias_followed_rm", "deception_rm"):
        for split in ("train", "eval"):
            name = f"{family}_{split}"
            pool_rows = read_jsonl(POOL_DIR / f"{name}.jsonl")
            for pool_row in pool_rows:
                row = by_key[(pool_row["question_id"], pool_row["condition"], pool_row["sample_idx"])]
                assert row["split"] == split and row["category"] == pool_row["category"], name
                if family == "deception_rm":
                    assert row["bias_rm"] == BIAS_OF_DECEPTION_SUBTYPE[pool_row["subtype"]], name
                else:
                    assert row["prompt"] == pool_row["prompt"] and row["completion"] == pool_row["completion"], name
                    assert row["bias_rm"] == pool_row["subtype"], name
                    assert row["bias_rm"] == ("bias" if pool_row["label"] == 1 else "no bias"), name
                    assert row["followed_cue"] == pool_row["followed_cue"], name
                    if family == "bias_followed_rm":
                        assert row["followed_cue"] is True, name
            results[name] = len(pool_rows)
            print(f"  GATE 260923 {name}: {len(pool_rows)} rows found with identical bias labels")
    return {"rows_matched": results, "pass": True}


def label_counts(rows: list[dict]) -> dict:
    out: dict = {}
    for split in ("train", "eval", "all"):
        out[split] = {}
        for condition in (*spec.CUED_CONDITIONS, "all"):
            selected = [
                row for row in rows
                if (split == "all" or row["split"] == split) and (condition == "all" or row["condition"] == condition)
            ]
            out[split][condition] = {
                "n": len(selected),
                "bias_rm": dict(Counter(row["bias_rm"] for row in selected)),
                "deception_rm": dict(Counter(row["deception_rm"] for row in selected)),
                "subtype": dict(Counter(deception_subtype(row) for row in selected)),
                "said_rm": dict(Counter(str(row["said_rm"]) for row in selected)),
                "said_sampled": dict(Counter(str(row["said_sampled"]) for row in selected)),
                "n_said_rm_differs_from_sampled": sum(
                    row["said_rm"] != row["said_sampled"] for row in selected if row["followup_text"] is not None
                ),
            }
    return out


BUILDERS = {
    "ref_cot": lambda gens_suffix: build_ref_cot(),
    "nocot_sampled": build_nocot_sampled,
    "nocot_logprob": build_nocot_logprob,
    "bio_cot": build_bio_cot,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True, choices=spec.ARMS)
    parser.add_argument("--gens-suffix", default="", help="read gens/<gen><suffix>/ (e.g. _dry)")
    parser.add_argument("--out-dir", default="arms", help="folder under the experiment dir to write arms/<arm>/ into")
    args = parser.parse_args()

    print(f"=== label.py --arm {args.arm} (spec {spec.spec_hash()}) ===")
    rows, none_answers = BUILDERS[args.arm](args.gens_suffix)
    rows.sort(key=lambda row: (row["question_id"], row["condition"], row["sample_idx"]))

    gates = {"partition": check_partition(rows)}
    if args.arm == "ref_cot":
        assert len(rows) == N_REF_ROWS, len(rows)
        gates["row_count"] = {"n_rows": len(rows), "expected": N_REF_ROWS, "pass": True}
        gates["against_260923_pools"] = check_against_260923_pools(rows)

    out_dir = spec.EXP_DIR / args.out_dir / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "labels.jsonl", rows)
    (out_dir / "none_answers.json").write_text(json.dumps(none_answers, indent=1))
    summary = {
        "spec_hash": spec.spec_hash(),
        "arm": args.arm,
        "gens_suffix": args.gens_suffix,
        "label_source": spec.LABEL_SOURCE[args.arm],
        "said_rule": f"p_yes_fp32 > {spec.SAID_THRESHOLD} -> Yes, < -> No, == -> null (score_pyes.py)",
        "n_rows": len(rows),
        "n_questions": len(none_answers),
        "gates": gates,
        "counts": label_counts(rows),
    }
    (out_dir / "labels_summary.json").write_text(json.dumps(summary, indent=1))
    counts_all = summary["counts"]["all"]["all"]
    print(f"  wrote {len(rows)} rows to {out_dir / 'labels.jsonl'}")
    print(f"  bias_rm {counts_all['bias_rm']}")
    print(f"  deception_rm {counts_all['deception_rm']}")
    print(f"  said_rm {counts_all['said_rm']} (thresholded P(Yes)); said_sampled {counts_all['said_sampled']}; "
          f"differ on {counts_all['n_said_rm_differs_from_sampled']} rows with a follow-up")


if __name__ == "__main__":
    main()
