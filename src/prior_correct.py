"""
Prior correction for the Chaturvedi classifier's output.

The classifier was trained on data that's 84.5% Hindu, 9.1% Muslim, 3.2%
Sikh, 2.45% Christian (real counts from REDS_train_multiclass.csv --
data/mappings/chaturvedi_train_class_prior.csv, computed once and saved
here since the raw REDS file isn't in this repo). When it's unsure, its
best statistical bet during training was "guess Hindu", so it tends to
default there. Standard fix (Saerens et al. 2002, "adjusting the outputs
of a classifier to new a priori probabilities"): divide each class's raw
probability by how often that class appeared in training, multiply by
whatever balance we actually want instead, renormalize.

    corrected(c) = raw(c) / train_prior(c) * target_prior(c)
    corrected(c) /= sum(corrected)   # renormalize back to summing to 1

Runs in the main venv -- takes a CSV of full per-class probabilities
already produced by chaturvedi_classify.py (which must run under
.venv-chaturvedi-lr; this script itself needs no sklearn).

IMPORTANT CAVEAT, worth restating every time this is used: this only
fixes the "defaults to the majority class when uncertain" failure mode.
It does NOT fix the other failure mode already found in this project --
the classifier being *specifically, confidently* wrong about a minority
class (moohammad/sulaiman -> Sikh). Boosting minority-class scores to fix
the first problem could make the second one worse, not better, since it
inflates exactly the scores that were already wrongly high. Always
re-check known error cases after applying this, never assume it helped.

Run:
    python src/prior_correct.py <input_full_probs.csv> <output.csv> [--target uniform|census]
"""
from __future__ import annotations

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPPINGS_DIR = os.path.join(REPO_ROOT, "data", "mappings")

RELIGIONS = ["Hindu", "Muslim", "Sikh", "Christian", "Jain", "Buddhist"]

# 2011 Census shares (approximate, for the "census" target option).
CENSUS_PRIOR = {"Hindu": 0.798, "Muslim": 0.142, "Sikh": 0.017, "Christian": 0.023, "Jain": 0.004, "Buddhist": 0.007}

# Equal weight across the 4 religions this project actually studies;
# Jain/Buddhist left at their real training share since we never select
# them for anything -- they just need to not artificially win ties.
UNIFORM_PRIOR = {"Hindu": 0.25, "Muslim": 0.25, "Sikh": 0.25, "Christian": 0.25, "Jain": None, "Buddhist": None}


def load_train_prior() -> dict:
    df = pd.read_csv(os.path.join(MAPPINGS_DIR, "chaturvedi_train_class_prior.csv"))
    return dict(zip(df["religion"], df["train_prior"]))


def correct(df: pd.DataFrame, target: str) -> pd.DataFrame:
    train_prior = load_train_prior()
    if target == "census":
        target_prior = CENSUS_PRIOR
    elif target == "uniform":
        target_prior = {r: (UNIFORM_PRIOR[r] if UNIFORM_PRIOR[r] is not None else train_prior[r]) for r in RELIGIONS}
    else:
        raise ValueError(f"Unknown target prior: {target}")

    df = df.copy()
    corrected = pd.DataFrame(index=df.index)
    for r in RELIGIONS:
        corrected[r] = df[f"prob_{r}"] / train_prior[r] * target_prior[r]
    row_sums = corrected.sum(axis=1)
    corrected = corrected.div(row_sums, axis=0)

    df["clf_predicted_religion_corrected"] = corrected.idxmax(axis=1)
    df["clf_max_prob_corrected"] = corrected.max(axis=1)
    return df


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python src/prior_correct.py <input_full_probs.csv> <output.csv> [--target uniform|census]", file=sys.stderr)
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]
    target = "uniform"
    if "--target" in sys.argv:
        target = sys.argv[sys.argv.index("--target") + 1]

    df = pd.read_csv(in_path)
    out = correct(df, target)
    out.to_csv(out_path, index=False)

    changed = (out["clf_predicted_religion"] != out["clf_predicted_religion_corrected"]).sum()
    print(f"Wrote {out_path} ({len(out)} rows, target prior = {target})")
    print(f"{changed}/{len(out)} predictions changed by correction")
    print("\nBefore -> after (only rows that changed):")
    diff = out[out["clf_predicted_religion"] != out["clf_predicted_religion_corrected"]]
    print(diff[["name", "clf_predicted_religion", "clf_max_prob", "clf_predicted_religion_corrected", "clf_max_prob_corrected"]].to_string(index=False))


if __name__ == "__main__":
    main()
