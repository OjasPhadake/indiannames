"""
Phase 3 -- religion labelling.

Primary labels come from hand-curated marker-name lists
(data/mappings/*_markers.csv), applied to the real frequency data from
Phase 1/2 -- following the same method Thorat & Attewell (2007) and
Banerjee, Bertrand, Datta & Mullainathan (2009) used. Counts are real;
category assignment is manual, not model-inferred, which is why every row
in name_bank.csv carries human_validated=False regardless of religion.

Chaturvedi & Chaturvedi's multiclass classifier (Hindu/Muslim/Sikh/
Christian/Jain/Buddhist) is layered on top as an independent cross-check
(src/chaturvedi_classify.py -> chaturvedi_predictions.csv, merged in below
as clf_scores/clf_agrees). Getting this working was not straightforward
and is worth recording:

  - The GitHub repo (RochanaChaturvedi/it-is-all-in-the-name) ships NO
    usable weights -- its notebook only ever loaded from the author's
    private Google Drive, and the repo's models/ directory contains a
    different task's CNN plus an orphaned SVM .sav with no vectorizer.
  - The actual trained weights turned out to be published all along, just
    not on GitHub: the paper's official CC0 replication data on Harvard
    Dataverse (DOI 10.7910/DVN/JOEVPN, its_all_in_the_name.zip, 1.4GB,
    found by Ojas). That archive contains real model_multiclass_*.sav /
    vectorizer_multiclass_*.sav pairs for both an SVM and a
    LogisticRegression variant.
  - Those pickles were made with scikit-learn 0.22.2.post1 (SVM) and
    1.0.2 (LR) respectively -- loading either under a modern sklearn
    (tested: 1.7.2) doesn't crash, it silently produces garbage: the
    TF-IDF transform breaks internally and the classifier degenerates to
    predicting the majority class ("Hindu") for nearly everything,
    including unambiguous names like Khan and Ansari. Confirmed by
    testing, not assumed -- this is the dangerous failure mode, not the
    version-mismatch warning itself.
  - Fixed by running chaturvedi_classify.py in a dedicated venv
    (.venv-chaturvedi-lr, Python 3.8 + numpy==1.21.6 + scipy==1.7.3 +
    scikit-learn==1.0.2) matching the LR pickle's actual version exactly.
    Chose LR over SVM: it's what predict_proba() requires for real
    probabilities (LinearSVC's decision_function() is a margin, not a
    probability, and the plan wants an actual probability threshold), and
    it was also more accurate in our own spot-check (malik, singh both
    correct under LR; both wrong under the version-matched SVM).
  - Real, confirmed model weaknesses (not version artifacts -- reproduced
    under the correctly-matched environment): "fernandes" predicts Hindu
    instead of Christian; "ayesha" is a near-coinflip between Muslim and
    Christian (0.509); Sikh given names in this dataset's Punjab rows
    (harapreeth, gurameeth, etc.) mostly predict Hindu, almost certainly
    because the classifier's own training data never saw this dataset's
    unusual Gurmukhi transliteration convention (implicit vowels
    preserved rather than dropped -- see sikh_firstname_markers.csv).

Net result: 83% agreement between the hand-curated marker lists and the
classifier across the 194 name_bank candidates both could score. See
DECISIONS.md #4 for the full writeup and CITATIONS.md for exact sources.

Sikh cell: now included, first names + surnames both, now that naampy
first names exist (Dataverse recovered 2026-09-06). The plan's own
requirement -- Singh/Kaur alone can't distinguish Sikh from Hindu Rajput/
the wider Hindi belt without pairing with a validated Punjabi Sikh given
name -- is implemented as two independently-built, independently-ranked
lists (a Punjab-anchored given-name list and the existing Singh/Kaur
surname list) meant to be paired when Phase 5 constructs synthetic full
names, not as a check that any real individual in the data has both
(the frequency tables don't carry per-person joint name pairs to check
that against). Many Punjabi Sikh given names are genuinely unisex in
practice (the "-preet"/"-jit"/"-deep" pattern) -- gender for these comes
from whatever the real electoral-roll data says, not an assumption.

Christian cell: anchored to Kerala and Goa markers only, both first names
and surnames. North-East Christian naming doesn't run through a
"distinctive name" pattern the way Kerala/Goa do (much more tribal/
ethnic-name-based), so it's a documented gap here rather than a guessed
marker list.

pranaam (the appeler org's Muslim/non-Muslim binary classifier) is still
not usable -- its HF-hosted weights aren't loadable by any released
version of the package (see CITATIONS.md). Not attempted here; the
Chaturvedi classifier's Muslim-vs-rest signal serves the same cross-check
purpose in the meantime.

See docs/PLAN.md Phase 3, DECISIONS.md #2 (caste held constant), #3 (hand
validation deferred) and #4 (classifier situation, now resolved for
Chaturvedi specifically).
"""
from __future__ import annotations

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
MAPPINGS_DIR = os.path.join(REPO_ROOT, "data", "mappings")
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")

CELL_TARGET = 25  # plan §3.3
CHRISTIAN_FREQ_FLOOR = 100  # lower than Phase 2's 500 -- Kerala/Goa are smaller
                             # populations than the national floor was tuned for,
                             # but still enough to exclude single-digit noise
SIKH_FREQ_FLOOR = 100        # same reasoning -- Punjab-anchored given names are a
                             # smaller pool than the national first-name floor

CHRISTIAN_ANCHOR_REGION = {"Kerala": "South", "Goa": "West"}
NAAMPY_RAW_PATH = os.path.join(RAW_DIR, "naampy_v2_raw.csv.gz")


def load_marker_list(filename: str, col: str = "name") -> set[str]:
    df = pd.read_csv(os.path.join(MAPPINGS_DIR, filename))
    return set(df[col].str.lower())


def load_clean_names(type_: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "clean_names.csv"))
    return df[df["type"] == type_]


def pooled_national(df: pd.DataFrame) -> pd.DataFrame:
    """Sum a name's n_total across regions for a nationally-pooled ranking."""
    return df.groupby("name")["n_total"].sum().reset_index().sort_values("n_total", ascending=False)


def pooled_national_with_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Same as pooled_national, but also carries a derived M/F label.

    clean_names.csv's prop_female is already gender-purity-filtered per
    region (Phase 2: kept only >=0.95 or <=0.05), so a weighted average
    across the regions a name survived in is a safe way to pool it to one
    national label without re-deriving purity from scratch.
    """
    weighted = df.assign(_wf=df["n_total"] * df["prop_female"])
    agg = weighted.groupby("name").agg(n_total=("n_total", "sum"), _wf=("_wf", "sum")).reset_index()
    agg["prop_female"] = agg["_wf"] / agg["n_total"]
    agg["gender"] = (agg["prop_female"] >= 0.5).map({True: "F", False: "M"})
    return agg.drop(columns=["_wf", "prop_female"]).sort_values("n_total", ascending=False)


def load_naampy_raw_filtered(states: set[str]) -> pd.DataFrame:
    """Streams the cached naampy raw file in chunks, keeping only rows for
    the given (lowercase, naampy's own short-form) state codes. Same
    memory concern as Phase 1's first-name aggregation -- 23.8M rows is
    too much to load whole, but filtering to 1-3 states first shrinks the
    kept rows enough to aggregate normally afterward."""
    kept = []
    for chunk in pd.read_csv(
        NAAMPY_RAW_PATH,
        usecols=["state", "first_name", "n_female", "n_male"],
        chunksize=2_000_000,
    ):
        sub = chunk[chunk["state"].isin(states)]
        if len(sub):
            kept.append(sub)
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=["state", "first_name", "n_female", "n_male"])


def build_muslim_cell(surnames: pd.DataFrame, firstnames: pd.DataFrame) -> pd.DataFrame:
    sn_markers = load_marker_list("muslim_surname_markers.csv", col="surname")
    sn_matched = surnames[surnames["name"].isin(sn_markers)]
    sn_pooled = pooled_national(sn_matched).head(CELL_TARGET)
    sn_pooled["type"] = "last"
    sn_pooled["gender"] = pd.NA

    fn_markers = load_marker_list("muslim_firstname_markers.csv")
    fn_matched = firstnames[firstnames["name"].isin(fn_markers)]
    fn_pooled = pooled_national_with_gender(fn_matched).head(CELL_TARGET)
    fn_pooled["type"] = "first"

    combined = pd.concat([sn_pooled, fn_pooled], ignore_index=True)
    combined["religion"] = "Muslim"
    combined["region"] = "ALL"
    combined["study"] = "primary_religion_gender"
    return combined


def build_hindu_cells(surnames: pd.DataFrame, firstnames: pd.DataFrame) -> pd.DataFrame:
    sn_markers = load_marker_list("hindu_upper_caste_surnames.csv", col="surname")
    sn_matched = surnames[surnames["name"].isin(sn_markers)]
    fn_markers = load_marker_list("hindu_firstname_markers.csv")
    fn_matched = firstnames[firstnames["name"].isin(fn_markers)]

    sn_pooled = pooled_national(sn_matched).head(CELL_TARGET)
    sn_pooled["type"] = "last"
    sn_pooled["gender"] = pd.NA
    fn_pooled = pooled_national_with_gender(fn_matched).head(CELL_TARGET)
    fn_pooled["type"] = "first"

    primary = pd.concat([sn_pooled, fn_pooled], ignore_index=True)
    primary["religion"] = "Hindu"
    primary["region"] = "ALL"
    primary["study"] = "primary_religion_gender"

    region_rows = []
    for region, group in sn_matched.groupby("region"):
        top = group.sort_values("n_total", ascending=False).head(CELL_TARGET)[["name", "n_total"]].copy()
        top["type"] = "last"
        top["gender"] = pd.NA
        top["religion"] = "Hindu"
        top["region"] = region
        top["study"] = "hindu_region_substudy"
        region_rows.append(top)
    for region, group in fn_matched.groupby("region"):
        top = pooled_national_with_gender(group).head(CELL_TARGET)
        top["type"] = "first"
        top["religion"] = "Hindu"
        top["region"] = region
        top["study"] = "hindu_region_substudy"
        region_rows.append(top)

    return pd.concat([primary] + region_rows, ignore_index=True)


def build_christian_cell() -> pd.DataFrame:
    # Surnames: instate raw, state-level proportions.
    sn_markers_df = pd.read_csv(os.path.join(MAPPINGS_DIR, "christian_surname_markers.csv"))
    instate_raw = pd.read_parquet(os.path.join(RAW_DIR, "instate_v2_surname_state_prop_raw.parquet"))

    sn_rows = []
    for anchor_state, group in sn_markers_df.groupby("anchor_state"):
        markers = set(group["surname"].str.lower())
        if anchor_state not in instate_raw.columns:
            print(f"WARNING: anchor state '{anchor_state}' not a column in the raw instate table -- skipping its markers", file=sys.stderr)
            continue
        sub = instate_raw[instate_raw["last_name"].isin(markers)][["last_name", anchor_state, "total_n"]].copy()
        sub["n_total"] = (sub[anchor_state] * sub["total_n"]).round()
        sub = sub[sub["n_total"] >= CHRISTIAN_FREQ_FLOOR].rename(columns={"last_name": "name"})
        sub["region"] = CHRISTIAN_ANCHOR_REGION[anchor_state]
        sub["type"] = "last"
        sub["gender"] = pd.NA
        sn_rows.append(sub[["name", "n_total", "region", "type", "gender"]])

    # First names: naampy raw, restricted to kerala/goa rows.
    fn_markers_df = pd.read_csv(os.path.join(MAPPINGS_DIR, "christian_firstname_markers.csv"))
    naampy_state_to_region = {"kerala": "South", "goa": "West"}
    naampy_raw = load_naampy_raw_filtered(set(naampy_state_to_region))

    fn_rows = []
    for anchor_state, group in fn_markers_df.groupby("anchor_state"):
        markers = set(group["name"].str.lower())
        sub = naampy_raw[(naampy_raw["state"] == anchor_state) & (naampy_raw["first_name"].isin(markers))].copy()
        sub = sub.groupby("first_name").agg(n_female=("n_female", "sum"), n_male=("n_male", "sum")).reset_index()
        sub["n_total"] = sub["n_female"] + sub["n_male"]
        sub = sub[sub["n_total"] >= CHRISTIAN_FREQ_FLOOR].rename(columns={"first_name": "name"})
        sub["gender"] = (sub["n_female"] / sub["n_total"] >= 0.5).map({True: "F", False: "M"})
        sub["region"] = naampy_state_to_region[anchor_state]
        sub["type"] = "first"
        fn_rows.append(sub[["name", "n_total", "region", "type", "gender"]])

    all_rows = sn_rows + fn_rows
    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=["name", "n_total", "region", "type", "gender"])
    combined = combined.sort_values("n_total", ascending=False).head(CELL_TARGET * 4)  # 2 anchor states x 2 name types
    combined["religion"] = "Christian"
    combined["study"] = "primary_religion_gender"
    return combined


def build_sikh_cell(surnames: pd.DataFrame) -> pd.DataFrame:
    # Surnames: Singh/Kaur, already real electoral-roll counts (surname table
    # has no per-state-restriction need here -- Singh/Kaur are nationally
    # common enough that Punjab-anchoring the surname would just throw away
    # most of the real signal; the given-name side below carries the
    # Punjab-specificity the plan's compound rule needs).
    sn_markers = load_marker_list("sikh_surname_candidates.csv", col="surname")
    sn_matched = surnames[surnames["name"].isin(sn_markers)]
    sn_pooled = pooled_national(sn_matched)
    sn_pooled["type"] = "last"
    sn_pooled["gender"] = sn_pooled["name"].map({"singh": "M", "kaur": "F"})

    # Given names: naampy raw, restricted to Punjab rows, per the plan's
    # explicit instruction to build this list "from naampy's Punjab table,
    # top-ranked, then hand-check" (docs/PLAN.md Phase 3 step 3).
    fn_markers = load_marker_list("sikh_firstname_markers.csv")
    naampy_raw = load_naampy_raw_filtered({"punjab"})
    fn_matched = naampy_raw[naampy_raw["first_name"].isin(fn_markers)].copy()
    fn_agg = fn_matched.groupby("first_name").agg(n_female=("n_female", "sum"), n_male=("n_male", "sum")).reset_index()
    fn_agg["n_total"] = fn_agg["n_female"] + fn_agg["n_male"]
    fn_agg = fn_agg[fn_agg["n_total"] >= SIKH_FREQ_FLOOR].rename(columns={"first_name": "name"})
    fn_agg["gender"] = (fn_agg["n_female"] / fn_agg["n_total"] >= 0.5).map({True: "F", False: "M"})
    fn_agg = fn_agg[["name", "n_total", "gender"]]
    fn_agg["type"] = "first"

    combined = pd.concat([sn_pooled[["name", "n_total", "type", "gender"]], fn_agg], ignore_index=True)
    combined["religion"] = "Sikh"
    combined["region"] = "ALL"
    combined["study"] = "primary_religion_gender"
    return combined


def main() -> None:
    surnames = load_clean_names("last")
    firstnames = load_clean_names("first")
    have_firstnames = len(firstnames) > 0
    if not have_firstnames:
        print("No first names in clean_names.csv yet -- Muslim/Hindu first-name cells, Christian first names, and the Sikh cell will all be skipped this run.", file=sys.stderr)

    print("Building Muslim cell (hand-curated markers)...")
    muslim = build_muslim_cell(surnames, firstnames)
    print(f"  {len(muslim)} names matched ({(muslim['type'] == 'last').sum()} surnames, {(muslim['type'] == 'first').sum()} first names)")

    print("Building Hindu cell + region sub-study (hand-curated upper-caste markers, DECISIONS.md #2)...")
    hindu = build_hindu_cells(surnames, firstnames)
    print(f"  {len(hindu)} rows (pooled + per-region, surnames + first names)")

    print("Building Christian cell (Kerala/Goa-anchored markers, real per-state frequency)...")
    christian = build_christian_cell()
    print(f"  {len(christian)} names matched ({(christian['type'] == 'last').sum()} surnames, {(christian['type'] == 'first').sum()} first names)")

    if have_firstnames:
        print("Building Sikh cell (Singh/Kaur surnames + Punjab-anchored given names, docs/PLAN.md Phase 3 step 3)...")
        sikh = build_sikh_cell(surnames)
        print(f"  {len(sikh)} names matched ({(sikh['type'] == 'last').sum()} surnames, {(sikh['type'] == 'first').sum()} first names)")
        bank_parts = [muslim, hindu, christian, sikh]
    else:
        print("Staging Sikh surname candidates only (NOT added to name_bank -- needs first names for the given-name compound rule)...")
        sikh_staged = pooled_national(surnames[surnames["name"].isin(load_marker_list("sikh_surname_candidates.csv", col="surname"))])
        staged_path = os.path.join(PROCESSED_DIR, "sikh_surname_candidates_staged.csv")
        sikh_staged.to_csv(staged_path, index=False)
        print(f"  Wrote {staged_path} ({len(sikh_staged)} rows)")
        bank_parts = [muslim, hindu, christian]

    bank = pd.concat(bank_parts, ignore_index=True)
    bank["source"] = "hand_curated_marker_list"
    bank["human_validated"] = False
    bank = bank.rename(columns={"n_total": "n"})

    clf_path = os.path.join(PROCESSED_DIR, "chaturvedi_predictions.csv")
    if os.path.exists(clf_path):
        clf = pd.read_csv(clf_path)
        bank = bank.merge(clf, on="name", how="left")
        bank["clf_scores"] = bank["clf_max_prob"]
        checked = bank["clf_predicted_religion"].notna()
        bank["clf_agrees"] = pd.NA
        bank.loc[checked, "clf_agrees"] = bank.loc[checked, "clf_predicted_religion"] == bank.loc[checked, "religion"]
        n_checked = bank["clf_predicted_religion"].notna().sum()
        n_agree = bank["clf_agrees"].sum()
        print(f"\nChaturvedi classifier cross-check: {n_agree}/{n_checked} marker-based labels agree with the classifier ({n_agree/n_checked:.0%})" if n_checked else "\nNo classifier predictions matched any name_bank candidates.")
        disagreements = bank[bank["clf_predicted_religion"].notna() & ~bank["clf_agrees"]]
        if len(disagreements):
            print("Disagreements (our marker label vs classifier's, worth a human look):")
            print(disagreements[["name", "type", "religion", "clf_predicted_religion", "clf_max_prob"]].to_string(index=False))
        bank = bank.drop(columns=["clf_predicted_religion"])
    else:
        bank["clf_scores"] = pd.NA
        bank["clf_agrees"] = pd.NA
        print(f"\nNo {clf_path} found -- run src/chaturvedi_classify.py under .venv-chaturvedi-lr first for a classifier cross-check (see that script's docstring for setup). Proceeding without it.", file=sys.stderr)

    bank = bank[["name", "type", "gender", "religion", "region", "study", "n", "source", "clf_scores", "clf_agrees", "human_validated"]]

    out_path = os.path.join(PROCESSED_DIR, "name_bank.csv")
    bank.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(bank)} rows)")
    print(bank.groupby(["religion", "type", "study"]).size().to_string())


if __name__ == "__main__":
    main()
