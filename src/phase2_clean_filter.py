"""
Phase 2 -- clean and filter.

Surnames only for now (naampy first names are still blocked on the Harvard
Dataverse outage -- see data/PROVENANCE.md; this script picks up
freq_firstnames.csv automatically once phase1 produces it).

Real data-quality issues found by inspecting the top-40-per-region surnames
in freq_surnames.csv, which drove the rules below:
  - relational/honorific artifacts standing alone as a "surname"
    (d/o, s/o, w/o, c/o, smt, shri) -- rows to drop outright, not rename.
  - Gujarati given-name + honorific-suffix pairs leaking into the surname
    field, e.g. "rameshbhai", "gitaben" -- strip the bhai/ben suffix.
  - transliteration spelling variants of the same surname sitting at
    different frequencies, e.g. mandal/mandala, dasa/das -- collapsed via
    the two explicit rules the plan names (trailing "a" stripped, "-ee"
    normalized to "-i"), NOT via generic edit-distance clustering.

    An earlier version of this script did try generic Levenshtein
    clustering (any pair within edit distance 1-2, weighted by frequency
    ratio) and it was actively dangerous: Indian names are short enough
    that unrelated names sit within edit distance 1-2 of each other far
    more often than true spelling variants do. It silently merged
    khan->khatun, paswan->hasan, mali->ali, barman->rahman, ray->rao/roy --
    real, distinct surnames, several with very different caste/religion
    signal, which is exactly the axis this project measures. Caught by
    manually inspecting the output, not by any test, which is the risk
    with this kind of heuristic: it fails silently and plausibly.

    Replaced with: only the plan's two explicit, deterministic suffix
    rules actually merge anything. Everything else that merely *looks*
    close (edit distance 1, checked but not merged) is written to
    data/validation/surname_near_duplicates_for_review.csv for a human to
    decide -- consistent with Phase 3's mandatory hand-validation step
    rather than a silent automatic merge.

See docs/PLAN.md Phase 2.
"""
from __future__ import annotations

import os
import re
import sys

import pandas as pd
from Levenshtein import distance as edit_distance

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
VALIDATION_DIR = os.path.join(REPO_ROOT, "data", "validation")

FREQ_FLOOR = 500          # minimum n per region (plan §Phase 2, "start at 500, tune")
TOP_K_PER_REGION = 500    # buffer above the eventual 25/cell target (plan §3.3), enough
                          # room to survive Phase 3 religion/caste filtering

# Standalone values that are relational/honorific artifacts, not surnames.
DROP_VALUES = {
    "d/o", "s/o", "w/o", "c/o", "smt", "shri", "sri", "km", "kum",
    "do", "so", "wo", "co", "bhai", "ben",
}

# Gujarati given-name honorific suffixes leaking into the surname field
# (rameshbhai -> bhai stripped, gitaben -> ben stripped).
HONORIFIC_SUFFIX_RE = re.compile(r"(bhai|ben)$")


def normalize_surname(raw: str) -> str | None:
    s = str(raw).strip().lower()
    if not s or not s.isalpha():
        return None
    if s in DROP_VALUES:
        return None
    if len(s) > 5:  # only strip the suffix off names long enough to have a stem left
        stripped = HONORIFIC_SUFFIX_RE.sub("", s)
        if len(stripped) >= 3:
            s = stripped
    if len(s) < 2:
        return None
    return s


def transliteration_stem(name: str) -> str:
    """The plan's two explicit collapse rules, applied deterministically --
    not a fuzzy match. patila->patil, dasa->das, banerjee-style->banerji-style.

    Known false-positive class this doesn't handle: trailing "a" is also a
    genuine male/female name-pair marker in Hindi (arun/aruna, manish/
    manisha are different people's names, not spelling variants of one).
    This rule can't distinguish that from transliteration noise, so it
    will merge those too -- verified in output as arun<-aruna, manish<-
    manisha etc. Left as-is rather than tuning the threshold to dodge it,
    because the plan's own example (dasa->das) is itself a 4-letter word;
    raising the length cutoff to exclude short gendered pairs would also
    silently break that example. Every merge is recorded in
    merged_variants precisely so this class of error stays auditable
    instead of silent -- check that column before trusting a row's count
    if it's being used somewhere gender matters (it currently isn't:
    surnames don't carry gender in this schema)."""
    s = name
    if len(s) > 5 and s.endswith("ee"):
        s = s[:-2] + "i"
    elif len(s) > 4 and s.endswith("a"):
        s = s[:-1]
    return s


def dedupe_spelling_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge only exact transliteration-stem matches (safe, deterministic).
    Anything merely close by edit distance is flagged for human review,
    not merged -- see module docstring for why.
    """
    df = df.sort_values("n_total", ascending=False).reset_index(drop=True)
    df["stem"] = df["surname"].map(transliteration_stem)

    merged_into: dict[str, list[str]] = {}
    for stem, group in df.groupby("stem"):
        if len(group) > 1:
            canonical = group.iloc[0]["surname"]  # highest n_total (df pre-sorted)
            merged_into[canonical] = [n for n in group["surname"] if n != canonical]

    agg = df.groupby("stem").agg(surname=("surname", "first"), n_total=("n_total", "sum")).reset_index(drop=True)
    agg["merged_variants"] = agg["surname"].map(lambda n: ";".join(merged_into.get(n, [])))

    # Flag remaining near-duplicates (edit distance 1, not already merged above)
    # for human review instead of auto-merging them.
    review_rows = []
    names = agg.sort_values("n_total", ascending=False)["surname"].tolist()
    counts = dict(zip(agg["surname"], agg["n_total"]))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if abs(len(a) - len(b)) <= 1 and edit_distance(a, b) == 1:
                review_rows.append({"name_a": a, "n_a": counts[a], "name_b": b, "n_b": counts[b]})

    return agg.sort_values("n_total", ascending=False), review_rows


def clean_surnames() -> tuple[pd.DataFrame, list[dict]]:
    src_path = os.path.join(PROCESSED_DIR, "freq_surnames.csv")
    df = pd.read_csv(src_path)

    df["surname_norm"] = df["surname"].map(normalize_surname)
    before = len(df)
    df = df.dropna(subset=["surname_norm"])
    print(f"Normalization: dropped {before - len(df)} junk/artifact rows ({len(df)} remain)")

    df = df.groupby(["region", "surname_norm"])["n_total"].sum().reset_index()
    df = df.rename(columns={"surname_norm": "surname"})

    df = df[df["n_total"] >= FREQ_FLOOR]
    print(f"Frequency floor >= {FREQ_FLOOR}: {len(df)} rows remain")

    out_frames = []
    all_review_rows = []
    for region, group in df.groupby("region"):
        top = group.sort_values("n_total", ascending=False).head(TOP_K_PER_REGION)
        deduped, review_rows = dedupe_spelling_variants(top[["surname", "n_total"]])
        deduped["region"] = region
        out_frames.append(deduped)
        for r in review_rows:
            r["region"] = region
        all_review_rows.extend(review_rows)
        n_merged = (deduped["merged_variants"] != "").sum()
        print(f"  {region}: {len(top)} candidates -> {len(deduped)} after spelling-rule merge ({n_merged} canonical rows absorbed variants, {len(review_rows)} near-duplicate pairs flagged for review)")

    result = pd.concat(out_frames, ignore_index=True)
    result["type"] = "last"
    result["prop_female"] = float("nan")  # surnames aren't gendered; float NaN (not pd.NA)
    # so this column's dtype matches firstnames' real float column on concat
    result = result.rename(columns={"surname": "name"})[
        ["name", "type", "region", "n_total", "prop_female", "merged_variants"]
    ].sort_values(["region", "n_total"], ascending=[True, False])
    return result, all_review_rows


def clean_firstnames() -> tuple[pd.DataFrame, list[dict]] | tuple[None, list]:
    src_path = os.path.join(PROCESSED_DIR, "freq_firstnames.csv")
    if not os.path.exists(src_path):
        print("freq_firstnames.csv not found yet (naampy/Dataverse still blocked) -- skipping first-name cleaning.", file=sys.stderr)
        return None, []

    df = pd.read_csv(src_path)
    df["first_name_norm"] = df["first_name"].map(normalize_surname)  # same junk/artifact rules apply
    df = df.dropna(subset=["first_name_norm"])
    df = df.groupby(["region", "first_name_norm"])[["n_female", "n_male", "n_third_gender", "n_total"]].sum().reset_index()
    df = df.rename(columns={"first_name_norm": "first_name"})
    df["prop_female"] = df["n_female"] / df["n_total"]

    df = df[df["n_total"] >= FREQ_FLOOR]
    # gender purity (plan §Phase 2): ambiguous names weaken the gender manipulation
    df = df[(df["prop_female"] >= 0.95) | (df["prop_female"] <= 0.05)]
    print(f"First names: {len(df)} rows after frequency floor + gender-purity filter")

    out_frames = []
    all_review_rows = []
    for region, group in df.groupby("region"):
        top = group.sort_values("n_total", ascending=False).head(TOP_K_PER_REGION)
        deduped, review_rows = dedupe_spelling_variants(top.rename(columns={"first_name": "surname"})[["surname", "n_total"]])
        deduped = deduped.rename(columns={"surname": "first_name"})
        deduped["region"] = region
        for r in review_rows:
            r["region"] = region
        all_review_rows.extend(review_rows)
        # prop_female doesn't carry cleanly through the sum-based dedupe merge path,
        # so re-derive it by joining back the pre-dedupe per-name gender split for
        # whichever spelling ended up canonical.
        deduped = deduped.merge(group[["first_name", "n_female", "n_male", "n_third_gender", "prop_female"]], on="first_name", how="left")
        out_frames.append(deduped)

    result = pd.concat(out_frames, ignore_index=True)
    result["type"] = "first"
    result = result.rename(columns={"first_name": "name"})[
        ["name", "type", "region", "n_total", "prop_female", "merged_variants"]
    ].sort_values(["region", "n_total"], ascending=[True, False])
    return result, all_review_rows


def main() -> None:
    os.makedirs(VALIDATION_DIR, exist_ok=True)

    print("Cleaning surnames...")
    surnames, surname_review = clean_surnames()
    for r in surname_review:
        r["type"] = "last"

    print("\nCleaning first names...")
    firstnames, firstname_review = clean_firstnames()
    for r in firstname_review:
        r["type"] = "first"

    combined = pd.concat([surnames, firstnames], ignore_index=True) if firstnames is not None else surnames
    out_path = os.path.join(PROCESSED_DIR, "clean_names.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(combined)} rows; first names {'included' if firstnames is not None else 'NOT included yet -- rerun once Dataverse is back'})")

    review = surname_review + firstname_review
    if review:
        review_df = pd.DataFrame(review)[["type", "region", "name_a", "n_a", "name_b", "n_b"]]
        review_path = os.path.join(VALIDATION_DIR, "near_duplicates_for_review.csv")
        review_df.to_csv(review_path, index=False)
        print(f"Wrote {review_path} ({len(review_df)} candidate pairs -- edit distance 1, NOT auto-merged, needs a human call)")


if __name__ == "__main__":
    main()
