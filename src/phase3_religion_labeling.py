"""
Phase 3 -- religion labelling (surnames only for now; see caveat below).

MAJOR DEVIATION FROM docs/PLAN.md, forced by tool availability, not choice:

  - Chaturvedi's multiclass classifier (Hindu/Muslim/Christian/Sikh/Jain/
    Buddhist) has NO published trained weights. Its notebook
    (RochanaChaturvedi/it-is-all-in-the-name, "Old code (deprecated)/
    Multiclass.ipynb") only ever loaded model files from the author's
    private Google Drive ("/content/drive/My Drive/name_to_religion/").
    The repo's models/ directory ships a *different* task's CNN (USA
    race, cnn_USA_*.h5) and a 2-class SVM .sav with no paired vectorizer
    file -- neither is usable. This isn't a temporary outage; the asset
    was simply never released. Permanently unavailable as designed.

  - pranaam (the Muslim/non-Muslim binary classifier) IS a real, current,
    installable package (0.9.0) -- but as of 2026-09-05 it downloads its
    model weights from Harvard Dataverse at runtime, same host that's
    504ing for naampy. Temporarily unavailable, tracked by the same
    recovery routine as first names (data/PROVENANCE.md).

With every classifier the plan named unavailable, Phase 3 today falls
back to exactly the method Thorat & Attewell (2007) and Banerjee, Bertrand,
Datta & Mullainathan (2009) used: explicit, hand-curated marker-surname
lists (data/mappings/*_surname*.csv), applied to the REAL frequency data
from Phase 1/2 -- so counts are real, but category assignment is manual,
not model-inferred. This is a stopgap, not a replacement: swap in pranaam
(and reconsider Chaturvedi's absence) once Dataverse recovers, and treat
every row this script writes as human_validated=False regardless of
religion, not just Christian/Sikh as DECISIONS.md #3 originally scoped --
the hand-validation gap now applies across the board.

Sikh cell: intentionally NOT included in name_bank.csv yet. The plan is
explicit that Singh/Kaur alone can't distinguish Sikh from Hindu Rajput/
the wider Hindi belt -- it requires pairing with a validated Punjabi Sikh
given name, and we don't have first names yet (naampy/Dataverse). Surname-
only candidates are staged separately for when that becomes possible.

Christian cell: anchored to Kerala and Goa surname markers only. North-
East Christian naming doesn't run through a "distinctive surname" pattern
the way Kerala/Goa do (much more tribal/ethnic-name-based), so it's a
documented gap here rather than a guessed marker list.

See docs/PLAN.md Phase 3, DECISIONS.md #2 (caste held constant) and #3
(hand validation deferred).
"""
from __future__ import annotations

import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
MAPPINGS_DIR = os.path.join(REPO_ROOT, "data", "mappings")
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")

CELL_TARGET = 25  # plan §3.3
CHRISTIAN_FREQ_FLOOR = 100  # lower than Phase 2's 500 -- Kerala/Goa are smaller
                             # populations than the national floor was tuned for,
                             # but still enough to exclude single-digit noise
                             # (e.g. a bare "fernandez" at n=31 slipping in
                             # because this cell reads the raw per-state table
                             # directly rather than clean_names.csv)

CHRISTIAN_ANCHOR_REGION = {"Kerala": "South", "Goa": "West"}


def load_marker_list(filename: str) -> set[str]:
    df = pd.read_csv(os.path.join(MAPPINGS_DIR, filename))
    return set(df["surname"].str.lower())


def load_clean_surnames() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "clean_names.csv"))
    return df[df["type"] == "last"]


def pooled_national(df: pd.DataFrame) -> pd.DataFrame:
    """Sum a name's n_total across regions for a nationally-pooled ranking."""
    return df.groupby("name")["n_total"].sum().reset_index().sort_values("n_total", ascending=False)


def build_muslim_cell(surnames: pd.DataFrame) -> pd.DataFrame:
    markers = load_marker_list("muslim_surname_markers.csv")
    matched = surnames[surnames["name"].isin(markers)]
    pooled = pooled_national(matched).head(CELL_TARGET)
    pooled["religion"] = "Muslim"
    pooled["region"] = "ALL"
    pooled["study"] = "primary_religion_gender"
    return pooled


def build_hindu_cells(surnames: pd.DataFrame) -> pd.DataFrame:
    markers = load_marker_list("hindu_upper_caste_surnames.csv")
    matched = surnames[surnames["name"].isin(markers)]

    pooled = pooled_national(matched).head(CELL_TARGET)
    pooled["religion"] = "Hindu"
    pooled["region"] = "ALL"
    pooled["study"] = "primary_religion_gender"

    region_rows = []
    for region, group in matched.groupby("region"):
        top = group.sort_values("n_total", ascending=False).head(CELL_TARGET)[["name", "n_total"]]
        top["religion"] = "Hindu"
        top["region"] = region
        top["study"] = "hindu_region_substudy"
        region_rows.append(top)

    return pd.concat([pooled] + region_rows, ignore_index=True)


def build_christian_cell() -> pd.DataFrame:
    markers_df = pd.read_csv(os.path.join(MAPPINGS_DIR, "christian_surname_markers.csv"))
    raw_path = os.path.join(RAW_DIR, "instate_v2_surname_state_prop_raw.parquet")
    raw = pd.read_parquet(raw_path)

    rows = []
    for anchor_state, group in markers_df.groupby("anchor_state"):
        markers = set(group["surname"].str.lower())
        state_col = anchor_state
        if state_col not in raw.columns:
            print(f"WARNING: anchor state '{anchor_state}' not a column in the raw instate table -- skipping its markers", file=os.sys.stderr)
            continue
        sub = raw[raw["last_name"].isin(markers)][["last_name", state_col, "total_n"]].copy()
        sub["n_in_state"] = (sub[state_col] * sub["total_n"]).round()
        sub = sub[sub["n_in_state"] >= CHRISTIAN_FREQ_FLOOR].rename(columns={"last_name": "name"})
        sub["region"] = CHRISTIAN_ANCHOR_REGION[anchor_state]
        sub["anchor_state"] = anchor_state
        rows.append(sub[["name", "n_in_state", "region", "anchor_state"]].rename(columns={"n_in_state": "n_total"}))

    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["name", "n_total", "region", "anchor_state"])
    combined = combined.sort_values("n_total", ascending=False).head(CELL_TARGET * 2)  # 2 anchor states, ~CELL_TARGET each
    combined["religion"] = "Christian"
    combined["study"] = "primary_religion_gender"
    return combined.drop(columns=["anchor_state"])


def stage_sikh_candidates(surnames: pd.DataFrame) -> pd.DataFrame:
    markers = load_marker_list("sikh_surname_candidates.csv")
    matched = surnames[surnames["name"].isin(markers)]
    return pooled_national(matched)


def main() -> None:
    surnames = load_clean_surnames()

    print("Building Muslim cell (hand-curated markers)...")
    muslim = build_muslim_cell(surnames)
    print(f"  {len(muslim)} surnames matched")

    print("Building Hindu cell + region sub-study (hand-curated upper-caste markers, DECISIONS.md #2)...")
    hindu = build_hindu_cells(surnames)
    print(f"  {len(hindu)} rows (pooled + per-region)")

    print("Building Christian cell (Kerala/Goa-anchored markers, real per-state frequency)...")
    christian = build_christian_cell()
    print(f"  {len(christian)} surnames matched")

    print("Staging Sikh candidates (NOT added to name_bank -- needs first-name compound validation)...")
    sikh_staged = stage_sikh_candidates(surnames)
    print(f"  {len(sikh_staged)} candidate surnames staged")

    bank = pd.concat([muslim, hindu, christian], ignore_index=True)
    bank["type"] = "last"
    bank["gender"] = pd.NA
    bank["source"] = "hand_curated_marker_list"
    bank["clf_scores"] = pd.NA
    bank["human_validated"] = False
    bank = bank.rename(columns={"n_total": "n"})
    bank = bank[["name", "type", "gender", "religion", "region", "study", "n", "source", "clf_scores", "human_validated"]]

    out_path = os.path.join(PROCESSED_DIR, "name_bank.csv")
    bank.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(bank)} rows)")
    print(bank.groupby(["religion", "study"]).size().to_string())

    staged_path = os.path.join(PROCESSED_DIR, "sikh_surname_candidates_staged.csv")
    sikh_staged.to_csv(staged_path, index=False)
    print(f"Wrote {staged_path} ({len(sikh_staged)} rows) -- for use once first names + given-name compound rule are available")


if __name__ == "__main__":
    main()
