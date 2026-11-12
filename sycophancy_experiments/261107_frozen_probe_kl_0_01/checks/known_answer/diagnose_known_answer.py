"""Per-answer differences of the known-answer test (seed 42): which answers differ most, and what is at their letter."""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/sycophancy-rl/sycophancy_setup")
import eval_checkpoint  # noqa: E402
import spec as S  # noqa: E402
from obfuscation_atlas.tasks.multiple_choice import find_answer_span  # noqa: E402

RUN = Path("/workspace/results/run_results/261104_iterative_kl_0_01/s42_round2_36lntjib")
ADAPTER = Path("/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s42_am1g59lu/checkpoints/final")
OUT = Path(sys.argv[1])

records = [json.loads(line) for line in (RUN / "checkpoints/responses/step_0.jsonl").read_text().splitlines()]
items = []
for r in records:
    completion = r["completion"].removesuffix("<|eot_id|>")
    letter, _, letter_index = find_answer_span(completion)
    items.append({"text": r["prompt"] + completion, "letter_char": len(r["prompt"]) + letter_index})
model, tokenizer = eval_checkpoint.load_hf_model(ADAPTER)
probes = {"penalty_probe": eval_checkpoint.load_probe(RUN, S.START_PROBE_SUBDIR)}
results = eval_checkpoint.score_followups(model, tokenizer, items, probes)
rows = []
for r, item, (_, _, _, p) in zip(records, items, results, strict=True):
    encoding = tokenizer(item["text"], add_special_tokens=False, return_offsets_mapping=True)
    position = eval_checkpoint.token_index_at(encoding["offset_mapping"], item["letter_char"])
    rows.append({
        "question_id": r["question_id"], "saved": r["probe_score"], "rescored": p["penalty_probe"],
        "abs_diff": abs(p["penalty_probe"] - r["probe_score"]), "tokens": len(encoding["input_ids"]),
        "letter_token": tokenizer.decode([encoding["input_ids"][position]]),
        "answer_end": r["completion"][-60:],
    })
rows.sort(key=lambda row: -row["abs_diff"])
OUT.write_text(json.dumps(rows, indent=1))
for row in rows[:8]:
    print(json.dumps(row))
print("rows over 0.02:", sum(row["abs_diff"] > 0.02 for row in rows), "over 0.01:", sum(row["abs_diff"] > 0.01 for row in rows), "of", len(rows))
