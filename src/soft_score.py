"""
Soft-scoring: rank every candidate name by a blended score instead of a
hard hand-list-or-nothing gate, per the design discussed with Ojas.

    score(name, religion) = w_hand * hand_indicator + (1 - w_hand) * clf_prob

hand_indicator is 1 if the name is on either hand list (ours or Ojas's)
for that religion, else 0. clf_prob is the classifier's RAW probability
(NOT prior-corrected -- src/prior_correct.py's own test found correction
doesn't reliably help and can actively hurt, so there's no reason to
inherit that risk here).

Names are ranked by this score within each religion (and gender, for
first names; region, for surnames), then cut at the same targets as
src/expand_corpus.py, and re-checked against the same known-error names
that broke the classifier-only and prior-correction experiments --
verifying this actually did what it was supposed to is the point,
not assuming it from the formula alone.

Run: python src/soft_score.py [--w-hand 0.7]
"""
from __future__ import annotations

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from expand_corpus import (  # noqa: E402
    FREQ_FLOOR, GENDER_HIGH, GENDER_LOW, REGIONS,
    load_user_lists, load_our_marker_lists, load_freq_tables, build_canonical_map,
)

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
TARGET_FIRSTNAME_PER_GENDER = 50
TARGET_SURNAME_PER_REGION = 25
TARGET_SIKH_SURNAME_POOLED = 100


def load_all_clf_scores() -> pd.DataFrame:
    """Union of every classifier run we've done so far, RAW probabilities
    (not prior-corrected). Keeps the widest possible name coverage."""
    parts = []
    for fname in ["chaturvedi_predictions.csv", "expand_corpus_predictions.csv", "broad_pool_predictions.csv"]:
        path = os.path.join(PROCESSED_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)[["name", "clf_predicted_religion", "clf_max_prob"]]
            parts.append(df)
    combined = pd.concat(parts, ignore_index=True)
    combined["name"] = combined["name"].str.lower()
    return combined.drop_duplicates(subset="name", keep="first")


def hand_indicator(name: str, religion: str, *dicts) -> int:
    for d in dicts:
        for key, names in d.items():
            if key[0] == religion and name in names:
                return 1
    return 0


def clf_prob_for(name: str, religion: str, clf: pd.DataFrame, clf_lookup: dict) -> float:
    row = clf_lookup.get(name)
    if row is None:
        return 0.0
    return row["clf_max_prob"] if row["clf_predicted_religion"] == religion else 0.0


def score_and_select(candidates: pd.DataFrame, name_col: str, religion: str, w_hand: float,
                      hand_dicts: list, clf_lookup: dict, target: int) -> pd.DataFrame:
    candidates = candidates.copy()
    candidates["hand"] = candidates[name_col].map(lambda n: hand_indicator(n, religion, *hand_dicts))
    candidates["clf"] = candidates[name_col].map(lambda n: clf_prob_for(n, religion, None, clf_lookup))
    candidates["score"] = w_hand * candidates["hand"] + (1 - w_hand) * candidates["clf"]
    return candidates.sort_values("score", ascending=False).head(target)


def main() -> None:
    w_hand = 0.7
    if "--w-hand" in sys.argv:
        w_hand = float(sys.argv[sys.argv.index("--w-hand") + 1])
    print(f"Soft-scoring with w_hand={w_hand} (classifier weight={1-w_hand:.1f})")

    user_fn, user_sn = load_user_lists()
    our_fn, our_sn = load_our_marker_lists()
    fn_freq, sn_freq = load_freq_tables()
    clf = load_all_clf_scores()
    clf_lookup = clf.set_index("name")[["clf_predicted_religion", "clf_max_prob"]].to_dict("index")

    rows = []
    print("\nFirst names:")
    for religion in ["Muslim", "Hindu", "Christian", "Sikh"]:
        pooled = fn_freq.groupby("first_name").agg(n_total=("n_total", "sum"), n_female=("n_female", "sum"), n_male=("n_male", "sum")).reset_index()
        pooled["prop_female"] = pooled["n_female"] / pooled["n_total"]
        pooled["gender"] = None
        pooled.loc[pooled["prop_female"] >= GENDER_HIGH, "gender"] = "F"
        pooled.loc[pooled["prop_female"] <= GENDER_LOW, "gender"] = "M"
        pooled = pooled[pooled["n_total"] >= FREQ_FLOOR]
        for gender in ["F", "M"]:
            pool = pooled[pooled["gender"] == gender]
            sel = score_and_select(pool, "first_name", religion, w_hand, [user_fn, our_fn], clf_lookup, TARGET_FIRSTNAME_PER_GENDER)
            n_clf_only = ((sel["hand"] == 0) & (sel["clf"] > 0)).sum()
            print(f"  {religion:10s} {gender}: {len(sel)}/{TARGET_FIRSTNAME_PER_GENDER} ({n_clf_only} classifier-only additions)")
            sel = sel.assign(religion=religion, region="ALL", type="first").rename(columns={"first_name": "name", "n_total": "n"})
            rows.append(sel[["name", "type", "gender", "religion", "region", "n", "hand", "clf", "score"]])

    print("\nSurnames:")
    for religion in ["Muslim", "Hindu", "Christian"]:
        for region in REGIONS:
            pool = sn_freq[sn_freq["region"] == region]
            sel = score_and_select(pool, "surname", religion, w_hand, [user_sn, our_sn], clf_lookup, TARGET_SURNAME_PER_REGION)
            n_clf_only = ((sel["hand"] == 0) & (sel["clf"] > 0)).sum()
            print(f"  {religion:10s} {region:6s}: {len(sel)}/{TARGET_SURNAME_PER_REGION} ({n_clf_only} classifier-only additions)")
            sel = sel.assign(religion=religion, region=region, type="last", gender=None).rename(columns={"surname": "name", "n_total": "n"})
            rows.append(sel[["name", "type", "gender", "religion", "region", "n", "hand", "clf", "score"]])

    pool = sn_freq.groupby("surname")["n_total"].sum().reset_index()
    sel = score_and_select(pool, "surname", "Sikh", w_hand, [user_sn, our_sn], clf_lookup, TARGET_SIKH_SURNAME_POOLED)
    n_clf_only = ((sel["hand"] == 0) & (sel["clf"] > 0)).sum()
    print(f"  {'Sikh':10s} pooled: {len(sel)}/{TARGET_SIKH_SURNAME_POOLED} ({n_clf_only} classifier-only additions)")
    clf_only_names = sel[(sel["hand"] == 0) & (sel["clf"] > 0)]["surname"].tolist()
    if clf_only_names:
        print(f"    classifier-only Sikh surnames added: {clf_only_names}")
    sel = sel.assign(religion="Sikh", region="ALL", type="last", gender=None).rename(columns={"surname": "name", "n_total": "n"})
    rows.append(sel[["name", "type", "gender", "religion", "region", "n", "hand", "clf", "score"]])

    out = pd.concat(rows, ignore_index=True)
    out_path = os.path.join(PROCESSED_DIR, f"soft_score_w{w_hand}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out)} rows)")

    # The actual test: did any of the already-confirmed error names get back in?
    known_bad = {"moohammad", "sulaiman", "moosa", "hussain", "abdullah", "ahammad", "naushad", "varun", "sen", "mohammad"}
    reintroduced = out[out["name"].isin(known_bad)]
    print(f"\nKnown-error names present in this output: {len(reintroduced)}")
    if len(reintroduced):
        print(reintroduced[["name", "religion", "hand", "clf", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
