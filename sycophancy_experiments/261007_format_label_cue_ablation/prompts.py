"""User content, prompt text and prompt ids per generation pass and condition.

Every prompt this experiment generates is derived from 260819's stored
`domain_questions.jsonl` renders, so that each arm differs from the reference
by exactly the one thing it tests:

  nocot   260819's user content with COT_SENTENCE replaced by NOCOT_SENTENCE
          (asserted: exactly one occurrence, nothing else changes); the cue
          sentence stays appended after the options; the assistant turn is
          prefilled with NOCOT_PREFILL.
  bio     the bio paragraph, a blank line, then 260819's no-cue content
          byte-for-byte (asserted), so 260819's no-cue rollouts are the exact
          removal counterfactual. No short cue sentence.

`question_row` is a row of domain_questions.jsonl.
"""

from __future__ import annotations

import json

from domain_imports import cue_text_for
from spec import (
    BIOS,
    COT_SENTENCE,
    CUED_CONDITIONS,
    DOMAIN_DIR,
    LETTERS,
    NOCOT_PREFILL,
    NOCOT_SENTENCE,
    PREFILL_LAST_TOKEN_ID,
    draw_bio_idx,
)


def load_questions() -> list[dict]:
    with (DOMAIN_DIR / "domain_questions.jsonl").open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 2290, f"expected 2290 questions, got {len(rows)}"
    return rows


def other_letter(letter: str) -> str:
    assert letter in LETTERS, letter
    return "B" if letter == "A" else "A"


def cue_letter(question_row: dict, condition: str) -> str | None:
    """The letter the cue names: the gold letter under cue_gold, the other under cue_dist."""
    if condition == "none":
        return None
    assert condition in CUED_CONDITIONS, condition
    gold = question_row["answer"]
    return gold if condition == "cue_gold" else other_letter(gold)


def short_cue_text(question_row: dict, condition: str) -> str | None:
    """260819's cue sentence for this question's drawn paraphrase (`random` variant)."""
    letter = cue_letter(question_row, condition)
    if letter is None:
        return None
    return cue_text_for(question_row["cue_template_idx"], letter)


def bio_text(question_row: dict, condition: str) -> str | None:
    letter = cue_letter(question_row, condition)
    if letter is None:
        return None
    bio = BIOS[draw_bio_idx(question_row["question_id"])]
    return bio["text"].format(subject=question_row["category"], letter=letter)


def nocot_user_content(question_row: dict, condition: str) -> str:
    base = question_row["user_content_none"]
    assert base.count(COT_SENTENCE) == 1, f"q{question_row['question_id']}: CoT sentence not found exactly once"
    content = base.replace(COT_SENTENCE, NOCOT_SENTENCE)
    cue = short_cue_text(question_row, condition)
    if cue is not None:
        content += "\n\n" + cue
        # the same replacement applied to 260819's stored cued render must agree:
        # the cue sentence is the only other thing in the prompt
        stored = question_row[f"user_content_{condition}"].replace(COT_SENTENCE, NOCOT_SENTENCE)
        assert content == stored, f"q{question_row['question_id']}/{condition}: cued no-CoT render disagrees with 260819's"
    return content


def bio_user_content(question_row: dict, condition: str) -> str:
    base = question_row["user_content_none"]
    bio = bio_text(question_row, condition)
    if bio is None:
        return base
    content = bio + "\n\n" + base
    assert content.split("\n\n", 1)[1] == base, "removing the bio paragraph must give the no-cue content"
    return content


def user_content(gen: str, question_row: dict, condition: str) -> str:
    if gen == "nocot":
        return nocot_user_content(question_row, condition)
    if gen == "bio":
        return bio_user_content(question_row, condition)
    if gen == "cot":
        # the reference: 260819's stored render, for the examples file only
        return question_row[f"user_content_{condition}"]
    raise ValueError(f"unknown gen {gen!r}")


def chat_prompt(tokenizer, content: str) -> str:
    """260819's `build_chat_prompt`: one user turn, generation prompt appended.
    `.rstrip()` as there (a few MMLU-Pro options carry trailing whitespace)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content.rstrip()}], tokenize=False, add_generation_prompt=True
    )


def prompt_text(tokenizer, gen: str, question_row: dict, condition: str) -> tuple[str, str]:
    """(prompt, prefill): the chat prompt and the assistant prefill the model
    continues from. `prefill` is "" except for the no-CoT pass. The model sees
    prompt + prefill; the stored turn-1 `completion` is prefill + what it
    generated (+ NOCOT_CLOSING), so prompt + completion is the full text."""
    prompt = chat_prompt(tokenizer, user_content(gen, question_row, condition))
    prefill = NOCOT_PREFILL if gen == "nocot" else ""
    return prompt, prefill


def prompt_ids(tokenizer, prompt: str, prefill: str) -> list[int]:
    """Token ids of what the model sees. add_special_tokens=False: the chat
    template already emits <|begin_of_text|>."""
    ids = tokenizer(prompt + prefill, add_special_tokens=False)["input_ids"]
    if prefill:
        assert ids[-1] == PREFILL_LAST_TOKEN_ID, f"prefill does not end in token {PREFILL_LAST_TOKEN_ID}: {ids[-5:]}"
    return ids
