"""
Phase 1 -- frequency base (region x gender).

Pulls the full underlying electoral-roll tables rather than using naampy's/
instate's single-name lookup/merge helpers (those are built for appending
stats onto an existing dataframe of names, not for dumping "top N names per
state").

naampy:  state, birth_year, first_name, n_female, n_male, n_third_gender
         Downloaded from Harvard Dataverse (naampy's own live source) at
         run time -- naampy itself never bundles this table.
         AS OF 2026-09-05: dataverse.harvard.edu is returning 504 on every
         endpoint checked (not just this file), which looks like a service
         outage rather than a dead link. Retry later; see PROVENANCE.md.

instate: last_name + one proportion column per state + total_n
         instate's *installed* pip package (0.1.7) hardcodes a download URL
         to a file that the upstream repo has since deleted from its `main`
         branch (a v3.0.0 rewrite on 2026-08-19 moved to a Hugging-Face
         hosted, single-surname "abstain" API and dropped the bulk table).
         That live URL now 404s.

         Pulled instead from the maintainers' own current official
         location: https://huggingface.co/gojiberries/instate (pinned
         commit, per that repo's README: "the package downloads this
         repository at an immutable commit so a released package cannot
         silently change models"). Verified byte-identical to the
         instate_unique_ln_state_prop_v2 table shipped in the old v2.0.0
         GitHub tag (same shape, same total_n sum) before switching to
         it -- this isn't new data, just a better-provenanced, non-Dataverse
         host for the same table. See PROVENANCE.md and CITATIONS.md.

See docs/PLAN.md Phase 1 and DECISIONS.md #1 for the state->region mapping
rationale.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPPING_PATH = os.path.join(REPO_ROOT, "data", "mappings", "state_region_mapping.csv")
OUT_DIR = os.path.join(REPO_ROOT, "data", "processed")
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")

NAAMPY_DATASET = "v2"  # 30 states, min 100 occurrences per name (see naampy CLI --help)

INSTATE_SURNAME_URL = (
    "https://huggingface.co/gojiberries/instate/resolve/main/"
    "instate_unique_ln_state_prop_v2.parquet"
)


def load_region_mapping() -> pd.DataFrame:
    mapping = pd.read_csv(MAPPING_PATH)
    mapping["state_norm"] = mapping["state"].str.strip().str.lower()
    return mapping


def build_firstname_table(mapping: pd.DataFrame) -> pd.DataFrame:
    from naampy.in_rolls_fn import InRollsFnData

    data_path = InRollsFnData.load_naampy_data(NAAMPY_DATASET)
    df = pd.read_csv(
        data_path,
        usecols=["state", "birth_year", "first_name", "n_female", "n_male", "n_third_gender"],
    )
    os.makedirs(RAW_DIR, exist_ok=True)
    df.to_csv(os.path.join(RAW_DIR, f"naampy_{NAAMPY_DATASET}_raw.csv.gz"), index=False, compression="gzip")

    agg = (
        df.groupby(["state", "first_name"])[["n_female", "n_male", "n_third_gender"]]
        .sum()
        .reset_index()
    )
    n_total = agg["n_female"] + agg["n_male"] + agg["n_third_gender"]
    agg["n_total"] = n_total
    agg["prop_female"] = agg["n_female"] / n_total

    agg["state_norm"] = agg["state"].str.strip().str.lower()
    merged = agg.merge(mapping[["state_norm", "region"]], on="state_norm", how="left")

    unmapped = merged[merged["region"].isna()]["state"].unique()
    if len(unmapped):
        print(f"WARNING: {len(unmapped)} naampy state value(s) not found in region mapping: {sorted(unmapped)}", file=sys.stderr)

    merged = merged.drop(columns=["state_norm"])
    return merged


def _download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"Using cached download: {dest}")
        return
    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)


def build_surname_region_table(mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Builds surname x region counts directly, without exploding to a
    surname x state long frame first (1.9M surnames x 34 states melted to
    ~63M rows OOM'd on an 8GB box). Since proportion*total_n summed over a
    region's state columns == total_n * (sum of that region's proportions),
    we can collapse each region's ~8-10 state columns down to one column
    per region while the frame is still wide, then melt only the resulting
    4 region columns.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    cache_path = os.path.join(RAW_DIR, "instate_v2_surname_state_prop_raw.parquet")
    _download(INSTATE_SURNAME_URL, cache_path)

    df = pd.read_parquet(cache_path)
    if "last_name" not in df.columns or "total_n" not in df.columns:
        raise RuntimeError(f"Unexpected instate schema, columns: {list(df.columns)}")

    state_cols = [c for c in df.columns if c not in ("last_name", "total_n")]
    for c in state_cols:
        df[c] = df[c].astype("float32")
    df["total_n"] = df["total_n"].astype("int64")

    state_to_region = dict(zip(mapping["state"], mapping["region"]))
    col_region = {c: state_to_region.get(c) for c in state_cols}
    unmapped = [c for c, r in col_region.items() if r is None]
    if unmapped:
        print(f"WARNING: {len(unmapped)} instate state column(s) not found in region mapping: {sorted(unmapped)}", file=sys.stderr)

    regions = sorted({r for r in col_region.values() if r is not None})
    out = pd.DataFrame({"surname": df["last_name"]})
    for region in regions:
        cols = [c for c, r in col_region.items() if r == region]
        prop_sum = df[cols].sum(axis=1)
        out[region] = (prop_sum * df["total_n"]).round().astype("int64")

    long = out.melt(id_vars="surname", value_vars=regions, var_name="region", value_name="n_total")
    long = long[long["n_total"] > 0]
    return long


def region_aggregate_firstnames(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.dropna(subset=["region"])
        .groupby(["region", "first_name"])[["n_female", "n_male", "n_third_gender", "n_total"]]
        .sum()
        .reset_index()
    )
    agg["prop_female"] = agg["n_female"] / agg["n_total"]
    return agg.sort_values(["region", "n_total"], ascending=[True, False])


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    mapping = load_region_mapping()

    print("Building surname x region table (gojiberries/instate on Hugging Face, pinned commit)...")
    sn_region = build_surname_region_table(mapping)
    sn_region = sn_region.sort_values(["region", "n_total"], ascending=[True, False])
    sn_out = os.path.join(OUT_DIR, "freq_surnames.csv")
    sn_region.to_csv(sn_out, index=False)
    print(f"Wrote {sn_out} ({len(sn_region)} rows)")

    print("Building first-name x state table from naampy (Harvard Dataverse)...")
    try:
        fn_state = build_firstname_table(mapping)
        fn_region = region_aggregate_firstnames(fn_state)
        fn_out = os.path.join(OUT_DIR, "freq_firstnames.csv")
        fn_region.to_csv(fn_out, index=False)
        print(f"Wrote {fn_out} ({len(fn_region)} rows)")
    except Exception as e:
        print(
            f"SKIPPED first-name extraction: {e}\n"
            "Harvard Dataverse has been returning 504 on every endpoint as of "
            "2026-09-05 (see PROVENANCE.md) -- this looks like a service outage, "
            "not a dead link. Re-run this script once it recovers; the surname "
            "table above is unaffected and already written.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
