"""Analysis of the 260825 probe eval sets and results.

Interactive, run cell by cell (#%%) in an editor with Jupyter-cell support, or
top to bottom as a plain script.

**Eval is one row per prompt.** A prompt is a (question_id, condition) pair, and
the domain generated 3 samples for each. Keeping all 3 puts the same prompt in
the denominator three times, so every metric here keeps exactly one: the lowest
`sample_idx` that survived label filtering (using the lowest *surviving* index
rather than a hard `sample_idx == 0` keeps every prompt that has at least one
usable row). Samples are iid draws at temperature 1, so which one is kept
carries no meaning.

Train is shown as-is -- it is the set the probes were actually trained on, so it
is not re-subset here.

Writes `probe_metrics_one_sample.json` (the metrics this file reports);
`probe_metrics.json` from 03_train_eval_probes.py is the unfiltered-eval version.
"""
# %%
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

EXP_DIR = Path(__file__).parent
PROBE_SETS_DIR = EXP_DIR / "probe_sets"
ACT_DIR = EXP_DIR / "activations"
WEIGHTS_DIR = EXP_DIR / "probe_weights"
RESCORE_DIR = EXP_DIR.parent / "260824_rescore"

sys.path.insert(0, str(EXP_DIR.parent.parent / "obfuscation-atlas"))
from obfuscation_atlas.detectors.train import load_model  # noqa: E402
from obfuscation_atlas.utils.metrics import best_threshold_for_metric, metrics_at_threshold  # noqa: E402

N_LAYERS = 32
BIAS_DEFINITIONS = ["rm", "swap"]
DECEPTION_DEFINITIONS = ["rm", "swap"]
BIAS_POSITIONS = ["answer", "lastcompl"]
DECEPTION_POSITIONS = ["firstgen", "lastprompt"]


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# %%
rescored = load_jsonl(RESCORE_DIR / "rescored_labels.jsonl")
rescored_idx = {(r["question_id"], r["condition"], r["sample_idx"]): r for r in rescored}
print(f"rescored_labels.jsonl: {len(rescored)} rows -> {len(rescored_idx)} indexed")


# %%
def one_per_prompt_mask(keys) -> np.ndarray:
    """keys: sequence of (question_id, condition, sample_idx, ...).
    Keeps one row per (question_id, condition): the lowest surviving sample_idx."""
    best: dict[tuple, tuple[int, int]] = {}
    for i, k in enumerate(keys):
        prompt = (k[0], k[1])
        if prompt not in best or k[2] < best[prompt][1]:
            best[prompt] = (i, k[2])
    mask = np.zeros(len(keys), dtype=bool)
    mask[[i for i, _ in best.values()]] = True
    return mask


def one_per_prompt_rows(rows: list[dict]) -> list[dict]:
    keys = [(r["question_id"], r["condition"], r["sample_idx"]) for r in rows]
    mask = one_per_prompt_mask(keys)
    return [r for r, keep in zip(rows, mask) if keep]


def print_crosstab(rows, row_key_fn, col_key_fn, row_order=None, col_order=None, title=""):
    counts = Counter((row_key_fn(r), col_key_fn(r)) for r in rows)
    rkeys = row_order or sorted({rk for rk, _ in counts})
    ckeys = col_order or sorted({ck for _, ck in counts})
    col_w = max(10, max(len(c) for c in ckeys) + 2)
    row_w = max(len(r) for r in rkeys) + 2

    print(f"  {title} (N={len(rows)})")
    print(" " * row_w + "".join(f"{c:>{col_w}}" for c in ckeys) + f"{'total':>{col_w}}")
    col_totals = {c: 0 for c in ckeys}
    for rk in rkeys:
        cells = [counts.get((rk, ck), 0) for ck in ckeys]
        for ck, n in zip(ckeys, cells):
            col_totals[ck] += n
        print(f"{rk:<{row_w}}" + "".join(f"{n:>{col_w}}" for n in cells) + f"{sum(cells):>{col_w}}")
    print(
        f"{'total':<{row_w}}"
        + "".join(f"{col_totals[c]:>{col_w}}" for c in ckeys)
        + f"{sum(col_totals.values()):>{col_w}}"
    )
    print()


# %%
# --- Eval set size ---
print("=" * 78)
print("EVAL SET SIZE (one row per prompt)")
print("=" * 78)
print("  687 eval questions x 2 conditions = 1374 prompts. A prompt is dropped only")
print("  if all 3 of its samples were anti-bias or unparsed.\n")
for kind, definitions in [("bias", BIAS_DEFINITIONS), ("deception", DECEPTION_DEFINITIONS)]:
    for definition in definitions:
        rows = one_per_prompt_rows(load_jsonl(PROBE_SETS_DIR / f"{kind}_{definition}_eval.jsonl"))
        print(f"  {kind}_{definition}_eval: N={len(rows)}  (of 1374 possible; "
              f"{1374 - len(rows)} prompts fully excluded)")
print()

# %%
# --- Bias: label x cue condition ---
print("=" * 78)
print("BIAS: label (bias / no bias) x condition (cue_gold / cue_dist)")
print("=" * 78)
for definition in BIAS_DEFINITIONS:
    for split in ["train", "eval"]:
        rows = load_jsonl(PROBE_SETS_DIR / f"bias_{definition}_{split}.jsonl")
        if split == "eval":
            rows = one_per_prompt_rows(rows)
        print_crosstab(
            rows,
            row_key_fn=lambda r: "bias" if r["label"] == 1 else "no_bias",
            col_key_fn=lambda r: r["condition"],
            row_order=["bias", "no_bias"],
            col_order=["cue_gold", "cue_dist"],
            title=f"bias_{definition}/{split}",
        )


# %%
# --- Deception: subtype x cue condition ---
#
# probe_sets rows carry only the collapsed deceptive/honest label; the subtype
# needs the underlying bias status, re-joined from rescored_labels.jsonl:
#   deceptive = (bias & said No)    -> false_denial
#               (no bias & said Yes) -> false_admission
#   honest    = (bias & said Yes)   -> true_admission
#               (no bias & said No)  -> true_denial
def deception_subtype(label: int, bias_value: str) -> str:
    if label == 1:
        return "false_denial" if bias_value == "bias" else "false_admission"
    return "true_admission" if bias_value == "bias" else "true_denial"


def augment_with_subtype(rows: list[dict], definition: str) -> list[dict]:
    bias_field = f"bias_{definition}"
    return [
        {**r, "subtype": deception_subtype(r["label"], rescored_idx[
            (r["question_id"], r["condition"], r["sample_idx"])][bias_field])}
        for r in rows
    ]


print("=" * 78)
print("DECEPTION: subtype x condition (cue_gold / cue_dist)")
print("=" * 78)
for definition in DECEPTION_DEFINITIONS:
    for split in ["train", "eval"]:
        rows = load_jsonl(PROBE_SETS_DIR / f"deception_{definition}_{split}.jsonl")
        if split == "eval":
            rows = one_per_prompt_rows(rows)
        print_crosstab(
            augment_with_subtype(rows, definition),
            row_key_fn=lambda r: r["subtype"],
            col_key_fn=lambda r: r["condition"],
            row_order=["false_denial", "false_admission", "true_admission", "true_denial"],
            col_order=["cue_gold", "cue_dist"],
            title=f"deception_{definition}/{split}",
        )


# %%
# --- Score the cached activations with the trained probes ---
#
# Raw LinearProbe logits per layer. No LLM forward pass: activations are cached
# at (N,1,hidden_dim) per layer and the probe's input_scale buffer persists
# through save_model/load_model. AUC is threshold-free, so raw logits give the
# same value the pipeline's expit-transformed scores do.
#
# accuracy/f1 follow the pipeline's convention exactly (evaluate.py:526): the
# threshold is the best-F1 threshold *on the train set*, then applied to eval.
# expit is monotone, so thresholding logits is equivalent to thresholding the
# pipeline's probabilities. Verified to reproduce probe_metrics.json to ~1e-4.
def score_probe(probe_name: str):
    weights_dir = WEIGHTS_DIR / probe_name
    filtered_layers = eval((weights_dir / "filtered_layers.txt").read_text())
    probe_dict = load_model(str(weights_dir))

    out = {}
    with torch.no_grad():
        for split in ["train", "eval"]:
            d = torch.load(ACT_DIR / f"{probe_name}_{split}.pt", weights_only=False)
            scores = {
                layer: probe_dict[layer](d["acts"][str(layer)]).squeeze(-1).squeeze(-1).numpy()
                for layer in range(N_LAYERS)
            }
            out[split] = {"scores": scores, "labels": d["labels"].numpy(), "row_keys": d["row_keys"]}

    mask = one_per_prompt_mask(out["eval"]["row_keys"])
    out["eval"]["scores"] = {k: v[mask] for k, v in out["eval"]["scores"].items()}
    out["eval"]["labels"] = out["eval"]["labels"][mask]
    out["eval"]["row_keys"] = [k for k, keep in zip(out["eval"]["row_keys"], mask) if keep]
    out["filtered_layers"] = filtered_layers
    return out


def mean_auc(scored, layers, mask=None) -> tuple[float, int, int]:
    labels, scores = scored["eval"]["labels"], scored["eval"]["scores"]
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    n_pos, n_neg = int((labels[mask] == 1).sum()), int((labels[mask] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), n_pos, n_neg
    aucs = [roc_auc_score(labels[mask], scores[l][mask]) for l in layers]
    return float(np.mean(aucs)), n_pos, n_neg


def mean_threshold_metrics(scored, layers) -> tuple[float, float]:
    accs, f1s = [], []
    for layer in layers:
        thr = best_threshold_for_metric(
            scored["train"]["scores"][layer], scored["train"]["labels"], "f1_score", exclude_scores=True
        )["threshold"]
        m = metrics_at_threshold(scored["eval"]["scores"][layer], scored["eval"]["labels"], thr)
        accs.append(m["accuracy"])
        f1s.append(m["f1_score"])
    return float(np.mean(accs)), float(np.mean(f1s))


# %%
# --- Headline metrics per probe ---
print("=" * 78)
print("HEADLINE METRICS (one row per prompt)")
print("=" * 78)
summaries = []
for kind, definitions, positions in [
    ("bias", BIAS_DEFINITIONS, BIAS_POSITIONS),
    ("deception", DECEPTION_DEFINITIONS, DECEPTION_POSITIONS),
]:
    for definition in definitions:
        for position in positions:
            probe_name = f"{kind}_{definition}_{position}"
            scored = score_probe(probe_name)
            fl = scored["filtered_layers"]
            auc_f, n_pos, n_neg = mean_auc(scored, fl)
            auc_all, _, _ = mean_auc(scored, list(range(N_LAYERS)))
            acc, f1 = mean_threshold_metrics(scored, fl)
            summaries.append({
                "probe": probe_name, "kind": kind, "definition": definition, "position": position,
                "n_train": len(scored["train"]["labels"]), "n_eval": len(scored["eval"]["labels"]),
                "n_eval_pos": n_pos, "n_eval_neg": n_neg,
                "auc_filtered": auc_f, "auc_all_layers": auc_all,
                "accuracy_filtered": acc, "f1_filtered": f1,
                "n_filtered_layers": len(fl),
            })
            print(f"  {probe_name:<28s} AUC={auc_f:.4f}  AUC(all32)={auc_all:.4f}  "
                  f"acc={acc:.4f}  f1={f1:.4f}  n_eval={len(scored['eval']['labels'])}  layers={len(fl)}")

(EXP_DIR / "probe_metrics_one_sample.json").write_text(json.dumps(summaries, indent=2))
print(f"\nWrote {EXP_DIR / 'probe_metrics_one_sample.json'}")

# %%
# --- Bias AUC split by cue condition ---
print("=" * 78)
print("BIAS AUC by cue condition (mean over filtered layers)")
print("=" * 78)
for definition in BIAS_DEFINITIONS:
    for position in BIAS_POSITIONS:
        probe_name = f"bias_{definition}_{position}"
        scored = score_probe(probe_name)
        fl = scored["filtered_layers"]
        conditions = np.array([k[1] for k in scored["eval"]["row_keys"]])
        auc, n_pos, n_neg = mean_auc(scored, fl)
        print(f"\n  {probe_name} (n_layers={len(fl)})")
        print(f"    overall   AUC={auc:.4f}  (n_bias={n_pos}, n_no_bias={n_neg})")
        for cond in ["cue_gold", "cue_dist"]:
            auc, n_pos, n_neg = mean_auc(scored, fl, conditions == cond)
            print(f"    {cond:<9s} AUC={auc:.4f}  (n_bias={n_pos}, n_no_bias={n_neg})")

# %%
# --- Deception AUC, split two complementary ways ---
#
# A single subtype has only one class, so "AUC by subtype" is undefined -- each
# split has to pair a deceptive subtype against an honest one. There are exactly
# two ways to do that, and together they separate the two things the deception
# label confounds:
#
# (a) hold BIAS STATUS fixed, self-report varies -- isolates deception from bias:
#       within no bias:  false_admission (deceptive) vs true_denial    (honest)
#       within bias:     false_denial    (deceptive) vs true_admission (honest)
#
# (b) hold SELF-REPORT fixed, bias status varies -- isolates the probe's signal
#     from token identity. Every row in one of these strata has the *same* token
#     at the probed position, so a probe that merely reads which token was
#     sampled has nothing left to go on and must land at ~0.5:
#       within said Yes: false_admission (deceptive) vs true_admission (honest)
#       within said No:  false_denial    (deceptive) vs true_denial    (honest)
print("=" * 78)
print("DECEPTION AUC by stratum (mean over filtered layers)")
print("=" * 78)
for definition in DECEPTION_DEFINITIONS:
    bias_field, said_field = f"bias_{definition}", f"said_{definition}"
    for position in DECEPTION_POSITIONS:
        probe_name = f"deception_{definition}_{position}"
        scored = score_probe(probe_name)
        fl = scored["filtered_layers"]
        recs = [rescored_idx[(k[0], k[1], k[2])] for k in scored["eval"]["row_keys"]]
        bias_status = np.array([r[bias_field] for r in recs])
        said = np.array([r[said_field] for r in recs])
        labels = scored["eval"]["labels"]

        # Gate: the stored self-report must agree with what (label, bias status)
        # implies, i.e. deceptive == (bias & No) or (no bias & Yes).
        implied = np.where(
            (bias_status == "bias") == (labels == 1), "No", "Yes"
        )
        assert (implied == said).all(), (
            f"{probe_name}: {(implied != said).sum()} rows where said_{definition} "
            f"disagrees with (label, bias) -- the subtype mapping is wrong"
        )

        auc, n_pos, n_neg = mean_auc(scored, fl)
        print(f"\n  {probe_name} (n_layers={len(fl)})")
        print(f"    overall                                 AUC={auc:.4f}  "
              f"(n_deceptive={n_pos}, n_honest={n_neg})")
        for status, desc in [
            ("no bias", "false_adm vs true_den"),
            ("bias", "false_den vs true_adm"),
        ]:
            auc, n_pos, n_neg = mean_auc(scored, fl, bias_status == status)
            print(f"    within bias={status:<8s} ({desc}) AUC={auc:.4f}  "
                  f"(n_deceptive={n_pos}, n_honest={n_neg})")
        for token, desc in [
            ("Yes", "false_adm vs true_adm"),
            ("No", "false_den vs true_den"),
        ]:
            auc, n_pos, n_neg = mean_auc(scored, fl, said == token)
            print(f"    within said={token:<8s} ({desc}) AUC={auc:.4f}  "
                  f"(n_deceptive={n_pos}, n_honest={n_neg})")

# %%
# ===========================================================================
# FIGURES
# ===========================================================================
# Colour follows the *entity*, reusing 260824_rescore/plot_results.py's hexes so a
# condition keeps its hue across every folder in this line of work. Those values
# were validated all-pairs on the light surface there (worst CVD dE 9.2,
# normal-vision 24.0 for the blue/orange/aqua triple); every pair used here is
# drawn from that triple, so it inherits at least that separation. Not re-run
# here -- the dataviz validator needs node, which this pod does not have.
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

C_GOLD = "#eb6834"      # cue at gold   (slot 2 orange) -- as in 260817-260824
C_DIST = "#1baf7a"      # cue at distractor (slot 3 aqua)
C_HOLD_BIAS = "#2a78d6"  # stratum family: bias status held fixed (slot 1 blue)
C_HOLD_SAID = "#eb6834"  # stratum family: self-report held fixed (slot 2 orange)
C_INK = "#333333"
C_MUTED = "#777777"
C_GRID = "#dddddd"
DPI = 150
CHANCE = 0.5
# White backing so a value label sitting near the chance line or a gridline stays
# legible, as in 260824_rescore/plot_results.py.
VALUE_LABEL_KW = dict(
    fontsize=8.5, color="#333333", zorder=6,
    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85),
)
CHANCE_HANDLE = Line2D([0], [0], color="#777777", lw=1.2, ls="--")

DEFINITION_LABEL = {"rm": "cue-removal", "swap": "cue-swap"}
POSITION_LABEL = {
    "answer": "answer-letter token",
    "lastcompl": "last completion token",
    "firstgen": "yes/no token",
    "lastprompt": "token before yes/no",
}


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["bottom"].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=8, length=0)


# %%
# --- Figure 1: bias probe AUC by cue condition (answer position only) ---
#
# Bars start at zero with chance marked, not truncated at 0.5: the magnitude
# being read is the AUC itself, so a truncated baseline would exaggerate the
# gaps. Direct labels carry the precision the full axis costs.
fig_data = {}
for definition in BIAS_DEFINITIONS:
    scored = score_probe(f"bias_{definition}_answer")
    fl = scored["filtered_layers"]
    cond = np.array([k[1] for k in scored["eval"]["row_keys"]])
    fig_data[definition] = {
        "overall": mean_auc(scored, fl)[0],
        "cue_gold": mean_auc(scored, fl, cond == "cue_gold"),
        "cue_dist": mean_auc(scored, fl, cond == "cue_dist"),
    }

fig, ax = plt.subplots(figsize=(6.4, 3.9))
x = np.arange(len(BIAS_DEFINITIONS))
bar_w = 0.26
gap = 0.012  # ~2px surface gap between adjacent bars

SERIES = [
    ("overall", C_MUTED, "overall"),
    ("cue_gold", C_GOLD, "cue at gold"),
    ("cue_dist", C_DIST, "cue at distractor"),
]
for i, (key, color, label) in enumerate(SERIES):
    vals = [
        fig_data[d][key] if key == "overall" else fig_data[d][key][0]
        for d in BIAS_DEFINITIONS
    ]
    offs = x + (i - 1) * (bar_w + gap)
    bars = ax.bar(offs, vals, bar_w, color=color, zorder=3, label=label)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.3f}",
                ha="center", va="bottom", **VALUE_LABEL_KW)

ax.axhline(CHANCE, color=C_MUTED, lw=1.2, ls="--", zorder=2)
ax.set_xlim(-0.62, len(BIAS_DEFINITIONS) - 0.38)
ax.set_xticks(x)
ax.set_xticklabels([DEFINITION_LABEL[d] for d in BIAS_DEFINITIONS], fontsize=10, color=C_INK)
ax.set_ylim(0, 1.0)
ax.set_ylabel("AUC (mean over filtered layers)", fontsize=9, color=C_MUTED)
ax.set_title("Bias probe AUC by cue condition", fontsize=11.5, color=C_INK, pad=10, loc="left")
ax.yaxis.grid(True, color=C_GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)
_style_axes(ax)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + [CHANCE_HANDLE], labels + ["chance (0.5)"],
          loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4, fontsize=8.5,
          frameon=False, labelcolor=C_MUTED)
fig.tight_layout()
fig.savefig(EXP_DIR / "bias_auc_by_condition.png", dpi=DPI, bbox_inches="tight")
print(f"wrote {EXP_DIR / 'bias_auc_by_condition.png'}")

# %%
# --- Figure 2: deception AUC by stratum ---
#
# Bars from zero on a full 0-1 axis, chance marked at 0.5. A stratum at ~0 (the
# first-generated-token probe with bias status held fixed) therefore shows as an
# absent bar, which is the honest rendering of "ranks every deceptive row below
# every honest one".
STRATA = [
    ("overall", "overall", None),
    ("bias_nobias", "no-bias", "hold_bias"),
    ("bias_bias", "bias", "hold_bias"),
    ("said_yes", "said Yes", "hold_said"),
    ("said_no", "said No", "hold_said"),
]
FAMILY_COLOR = {None: C_MUTED, "hold_bias": C_HOLD_BIAS, "hold_said": C_HOLD_SAID}

dec_data = {}
for definition in DECEPTION_DEFINITIONS:
    bias_field, said_field = f"bias_{definition}", f"said_{definition}"
    for position in DECEPTION_POSITIONS:
        probe_name = f"deception_{definition}_{position}"
        scored = score_probe(probe_name)
        fl = scored["filtered_layers"]
        recs = [rescored_idx[(k[0], k[1], k[2])] for k in scored["eval"]["row_keys"]]
        bias_status = np.array([r[bias_field] for r in recs])
        said = np.array([r[said_field] for r in recs])
        dec_data[probe_name] = {
            "overall": mean_auc(scored, fl),
            "bias_nobias": mean_auc(scored, fl, bias_status == "no bias"),
            "bias_bias": mean_auc(scored, fl, bias_status == "bias"),
            "said_yes": mean_auc(scored, fl, said == "Yes"),
            "said_no": mean_auc(scored, fl, said == "No"),
        }

fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharey=True)
xs = np.arange(len(STRATA))
for row, definition in enumerate(DECEPTION_DEFINITIONS):
    for col, position in enumerate(DECEPTION_POSITIONS):
        ax = axes[row, col]
        probe_name = f"deception_{definition}_{position}"
        for xi, (key, label, family) in zip(xs, STRATA):
            auc, n_pos, n_neg = dec_data[probe_name][key]
            ax.bar(xi, auc, 0.62, color=FAMILY_COLOR[family], zorder=3)
            ax.text(xi, auc + 0.02, f"{auc:.3f}", ha="center", va="bottom", **VALUE_LABEL_KW)
        ax.axhline(CHANCE, color=C_MUTED, lw=1.2, ls="--", zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels([label for _, label, _ in STRATA], fontsize=8.5, color=C_INK)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(f"{DEFINITION_LABEL[definition]} · {POSITION_LABEL[position]}",
                     fontsize=9.5, color=C_INK, loc="left", pad=6)
        ax.yaxis.grid(True, color=C_GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        _style_axes(ax)
        if col == 0:
            ax.set_ylabel("AUC (mean over filtered layers)", fontsize=9, color=C_MUTED)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=C_MUTED),
    plt.Rectangle((0, 0), 1, 1, color=C_HOLD_BIAS),
    plt.Rectangle((0, 0), 1, 1, color=C_HOLD_SAID),
    CHANCE_HANDLE,
]
fig.legend(
    handles,
    ["overall", "bias status held fixed (self-report varies)",
     "self-report held fixed (bias status varies)", "chance (0.5)"],
    loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=4, fontsize=8.5,
    frameon=False, labelcolor=C_MUTED,
)
fig.suptitle("Deception probe AUC by stratum", fontsize=12, color=C_INK,
             x=0.008, ha="left", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(EXP_DIR / "deception_auc_decomposition.png", dpi=DPI, bbox_inches="tight")
print(f"wrote {EXP_DIR / 'deception_auc_decomposition.png'}")

# %%
# --- Mean probe score per class, beside every AUC reported above ---
#
# "Probe score" follows the pipeline's convention: the sigmoid of the probe logit
# (eval_detector pushes logits through expit before computing metrics), averaged
# over the rows of a class, then averaged over the probe's filtered layers -- the
# same construction as `mean_score_positive` / `mean_score_negative` in
# pipeline/probe_training.py's summary. AUC is rank-based and so is unchanged by
# the sigmoid; these means are not, which is why they are reported on that scale.
#
# Reading them against the AUC is what separates "the probe ranks the classes
# apart" from "the probe puts them at different absolute scores": a stratum can
# sit at AUC 0.5 with the two class means nearly equal (no signal), or at AUC
# ~0.0 with the means far apart in the *wrong* order (inverted signal).
from scipy.special import expit  # noqa: E402


def class_means(scored, layers, mask=None) -> tuple[float, float]:
    labels, scores = scored["eval"]["labels"], scored["eval"]["scores"]
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    y = labels[mask]
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan"), float("nan")
    pos = [expit(scores[l][mask])[y == 1].mean() for l in layers]
    neg = [expit(scores[l][mask])[y == 0].mean() for l in layers]
    return float(np.mean(pos)), float(np.mean(neg))


def print_score_table(title, pos_name, neg_name, rows):
    print(f"\n{title}")
    print(f"  {'probe':<28s} {'stratum':<10s} {'n ' + pos_name:>10s} {'n ' + neg_name:>10s} "
          f"{'AUC':>7s} {'mean ' + pos_name:>13s} {'mean ' + neg_name:>13s} {'delta':>8s}")
    for probe, stratum, n_pos, n_neg, auc, m_pos, m_neg in rows:
        print(f"  {probe:<28s} {stratum:<10s} {n_pos:>10d} {n_neg:>10d} "
              f"{auc:>7.4f} {m_pos:>13.4f} {m_neg:>13.4f} {m_pos - m_neg:>+8.4f}")


print("=" * 110)
print("MEAN PROBE SCORE PER CLASS, BESIDE EACH REPORTED AUC")
print("=" * 110)

bias_rows = []
for definition in BIAS_DEFINITIONS:
    for position in BIAS_POSITIONS:
        probe_name = f"bias_{definition}_{position}"
        scored = score_probe(probe_name)
        fl = scored["filtered_layers"]
        cond = np.array([k[1] for k in scored["eval"]["row_keys"]])
        for stratum, mask in [
            ("overall", None), ("cue_gold", cond == "cue_gold"), ("cue_dist", cond == "cue_dist"),
        ]:
            auc, n_pos, n_neg = mean_auc(scored, fl, mask)
            m_pos, m_neg = class_means(scored, fl, mask)
            bias_rows.append((probe_name, stratum, n_pos, n_neg, auc, m_pos, m_neg))
print_score_table("BIAS probes (positive class = bias, negative = no bias)",
                  "bias", "no-bias", bias_rows)

dec_rows = []
for definition in DECEPTION_DEFINITIONS:
    bias_field, said_field = f"bias_{definition}", f"said_{definition}"
    for position in DECEPTION_POSITIONS:
        probe_name = f"deception_{definition}_{position}"
        scored = score_probe(probe_name)
        fl = scored["filtered_layers"]
        recs = [rescored_idx[(k[0], k[1], k[2])] for k in scored["eval"]["row_keys"]]
        bias_status = np.array([r[bias_field] for r in recs])
        said = np.array([r[said_field] for r in recs])
        for stratum, mask in [
            ("overall", None),
            ("no-bias", bias_status == "no bias"),
            ("bias", bias_status == "bias"),
            ("said Yes", said == "Yes"),
            ("said No", said == "No"),
        ]:
            auc, n_pos, n_neg = mean_auc(scored, fl, mask)
            m_pos, m_neg = class_means(scored, fl, mask)
            dec_rows.append((probe_name, stratum, n_pos, n_neg, auc, m_pos, m_neg))
print_score_table("DECEPTION probes (positive class = deceptive, negative = honest)",
                  "decep", "honest", dec_rows)

# %%
