"""
"What if we just let the classifier pick ~100 names per religion on its
own, with no hand list at all -- how much would that overlap with the
hand-list-backed corpus (name_bank_expanded.csv)?" Ojas asked this
directly; this is that experiment, extracted into a real script instead
of the one-off analysis it started as.

Uses data/processed/broad_pool_predictions.csv (the ~5,650 most common
first names and surnames nationally, already classified -- see
src/chaturvedi_classify.py and DECISIONS.md #5/#6 for how that pool was
built) and applies the same selection rule as expand_corpus.py --
same real frequency ranking, same targets -- except membership is decided
by "the classifier's top pick is this religion" instead of "on a hand
list", so the two outputs are comparable apples-to-apples.

Result (documented in the conversation and DECISIONS.md): first names
overlap almost not at all (0-7 out of ~50 per cell) -- the two methods
answer different questions, hand lists pick names a human recognizes as
*distinctively* religious, the classifier just tags whatever's already
popular. Surnames overlap partially (Muslim best, ~30-45%). The
classifier-only Sikh surname list reproduces the exact same errors found
elsewhere in this project (moosa, sulaiman, hussain, abdullah as "Sikh").

Run: python src/classifier_only_comparison.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from phase2_clean_filter import normalize_surname  # noqa: E402
from expand_corpus import build_canonical_map, REGIONS  # noqa: E402

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
TARGET_FIRSTNAME_PER_GENDER = 50
TARGET_SURNAME_PER_REGION = 25
TARGET_SIKH_SURNAME_POOLED = 100


def build_national_freq_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    fn = pd.read_csv(os.path.join(PROCESSED_DIR, "freq_firstnames.csv"))
    fn["first_name"] = fn["first_name"].map(normalize_surname)
    fn = fn.dropna(subset=["first_name"])
    fn_nat = fn.groupby("first_name")[["n_female", "n_male", "n_total"]].sum()
    cmap = build_canonical_map(fn_nat["n_total"])
    fn_nat = fn_nat.rename(index=cmap).groupby(level=0).sum()
    fn_nat["prop_female"] = fn_nat["n_female"] / fn_nat["n_total"]
    fn_nat["gender"] = None
    fn_nat.loc[fn_nat["prop_female"] >= 0.6, "gender"] = "F"
    fn_nat.loc[fn_nat["prop_female"] <= 0.4, "gender"] = "M"

    sn = pd.read_csv(os.path.join(PROCESSED_DIR, "freq_surnames.csv"))
    sn["surname"] = sn["surname"].map(normalize_surname)
    sn = sn.dropna(subset=["surname"])
    sn_region = sn.groupby(["region", "surname"])["n_total"].sum().reset_index()
    sn_nat_totals = sn.groupby("surname")["n_total"].sum()
    cmap2 = build_canonical_map(sn_nat_totals)
    sn_region["surname"] = sn_region["surname"].map(cmap2)
    sn_region = sn_region.groupby(["region", "surname"])["n_total"].sum().reset_index()
    return fn_nat, sn_region


def main() -> None:
    fn_nat, sn_region = build_national_freq_tables()
    clf = pd.read_csv(os.path.join(PROCESSED_DIR, "broad_pool_predictions.csv"))
    clf["name"] = clf["name"].str.lower()
    expanded = pd.read_csv(os.path.join(PROCESSED_DIR, "name_bank_expanded.csv"))

    print("=== Classifier-only first names vs. hand-list-backed corpus ===")
    rows = []
    for rel in ["Muslim", "Hindu", "Christian", "Sikh"]:
        hits = set(clf[clf["clf_predicted_religion"] == rel]["name"])
        for gender in ["F", "M"]:
            pool = fn_nat[fn_nat.index.isin(hits) & (fn_nat["gender"] == gender)].sort_values("n_total", ascending=False).head(TARGET_FIRSTNAME_PER_GENDER)
            ours = set(expanded[(expanded["religion"] == rel) & (expanded["type"] == "first") & (expanded["gender"] == gender)]["name"])
            overlap = set(pool.index) & ours
            print(f"  {rel:10s} {gender}: classifier-only picked {len(pool)}, overlap with hand-list corpus: {len(overlap)}/{len(ours)}")
            rows.append({"religion": rel, "type": "first", "gender": gender, "clf_only_picked": len(pool), "overlap": len(overlap), "hand_list_total": len(ours)})

    print("\n=== Classifier-only surnames vs. hand-list-backed corpus ===")
    for rel in ["Muslim", "Hindu", "Christian"]:
        hits = set(clf[clf["clf_predicted_religion"] == rel]["name"])
        for region in REGIONS:
            pool = sn_region[(sn_region["region"] == region) & (sn_region["surname"].isin(hits))].sort_values("n_total", ascending=False).head(TARGET_SURNAME_PER_REGION)
            ours = set(expanded[(expanded["religion"] == rel) & (expanded["type"] == "last") & (expanded["region"] == region)]["name"])
            overlap = set(pool["surname"]) & ours
            print(f"  {rel:10s} {region:6s}: classifier-only picked {len(pool)}, overlap: {len(overlap)}/{len(ours)}")
            rows.append({"religion": rel, "type": "last", "region": region, "clf_only_picked": len(pool), "overlap": len(overlap), "hand_list_total": len(ours)})

    hits = set(clf[clf["clf_predicted_religion"] == "Sikh"]["name"])
    pool = sn_region.groupby("surname")["n_total"].sum()
    pool = pool[pool.index.isin(hits)].sort_values(ascending=False).head(TARGET_SIKH_SURNAME_POOLED)
    ours = set(expanded[(expanded["religion"] == "Sikh") & (expanded["type"] == "last")]["name"])
    overlap = set(pool.index) & ours
    print(f"  {'Sikh':10s} pooled: classifier-only picked {len(pool)}, overlap: {len(overlap)}/{len(ours)}")
    print(f"    classifier-only Sikh surname picks: {sorted(pool.index.tolist())}")
    rows.append({"religion": "Sikh", "type": "last", "region": "ALL", "clf_only_picked": len(pool), "overlap": len(overlap), "hand_list_total": len(ours)})

    out_path = os.path.join(PROCESSED_DIR, "classifier_only_vs_handlist_overlap.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
