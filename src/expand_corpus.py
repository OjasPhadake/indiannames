"""
Builds an expanded name corpus (~100 first names + ~100 surnames per
religion) for a downstream use case that needs volume over the tight
~25/cell target in docs/PLAN.md's original design. Merges three sources
per Ojas's explicit request ("best of all 3"):

  1. Ojas's own hand-typed FIRST_NAME_BANK / REGION_SURNAME_MAP
     (data/mappings/user_provided_names_raw.py).
  2. This project's hand-curated marker lists (data/mappings/*_markers.csv).
  3. The Chaturvedi classifier's predictions (statistically independent of
     both hand lists).

A name from source 1 or 2 counts as a *candidate* for a religion; real
electoral-roll frequency (from freq_firstnames.csv / freq_surnames.csv --
Phase 1's output, deliberately used instead of clean_names.csv because
Phase 2's frequency floor + top-500-per-region cut is exactly the kind of
"gate" being lowered here) then decides ranking and which candidates
actually clear a (lowered) floor. Nothing is included on hand-list say-so
alone with zero real frequency behind it.

Source 3 (the classifier) ended up NOT being used as a candidate source,
despite that being the original plan -- see the threshold note below for
why. It's still run and merged in as clf_predicted_religion/clf_max_prob
on every row, exactly like phase3_religion_labeling.py does, so you can
see where the classifier agrees or disagrees with the hand-list-backed
selection.

Two-pass because the classifier needs its own Python/sklearn environment
(see chaturvedi_classify.py's docstring):

    python src/expand_corpus.py prepare
    # -> writes data/processed/expand_corpus_names_to_classify.txt
    source .venv-chaturvedi-lr/bin/activate
    python src/chaturvedi_classify.py data/processed/expand_corpus_names_to_classify.txt data/processed/expand_corpus_predictions.csv
    deactivate  # or open a new shell back in .venv
    source .venv/bin/activate
    python src/expand_corpus.py finalize
    # -> writes data/processed/name_bank_expanded.csv

Lowered gates, explicit about what changed vs. the strict pipeline
(src/phase3_religion_labeling.py, which is untouched by this script and
remains the plan's primary, tighter-validated output):

  - Frequency floor: 30 (was 500 in Phase 2) -- still excludes single-
    or double-digit noise, but admits real regional/community names that
    the strict floor would cut.
  - Gender purity: prop_female >= 0.6 -> F, <= 0.4 -> M (was >=0.95/<=0.05).
    Names in the 0.4-0.6 band are excluded as genuinely ambiguous rather
    than forced into a bucket -- a resume name-swap study needs the
    gender signal to actually be clean.
  - Classifier-only candidates (a name with no hand-list backing at all)
    are NOT included, at any threshold -- tried 0.5 first, then 0.85 after
    finding real problems, and 0.85 still let confirmed errors through:
    "moohammad" (a spelling variant, 90.7% confidence) and "sulaiman"/
    "moosa" (clearly Muslim names) as *Sikh surnames* -- the same
    mohammad->Sikh bias already documented in DECISIONS.md #4, just
    resurfacing on a different spelling each time the threshold moved --
    and "varun"/"sen" (Hindu-pattern names) as *Christian surnames* at
    90%+ confidence too. This isn't a threshold-tuning problem; the
    classifier has a specific, repeatable bias that no single cutoff
    reliably screens out. Tested disabling classifier-only entirely: hand
    lists alone (source 1 + 2) already reach 94-100/religion for
    Muslim/Hindu/Christian via real-frequency ranking, so the safety
    trade was cheap. Sikh stays genuinely thinner (~55-60), honestly, per
    DECISIONS.md #4/#5 -- not padded with classifier guesses to hit 100.

  - Applies the same transliteration-stem merge Phase 2 uses (trailing
    "a" stripped, "-ee" normalized to "-i") to freq_firstnames.csv /
    freq_surnames.csv before ranking -- neither file has been through
    Phase 2's own dedup (this script deliberately reads Phase 1's less-
    filtered output instead). Without it, doubled-vowel transliteration
    variants of ordinary names (rekhaa, meeraa, sulekhaa -- all real
    spellings of Rekha/Meera/Sulekha in this dataset) surfaced as
    separate "new" low-frequency candidates that the classifier handles
    badly on the small character sample. For first names specifically,
    the merge is done *within* the already-computed gender bucket, not
    before -- arun/aruna and manish/manisha are a real male/female name
    pair, not one person's misspelling (see transliteration_stem's own
    docstring), and merging them pre-gender would corrupt that.

See DECISIONS.md #5 for why this exists as a second output rather than
replacing name_bank.csv, and CITATIONS.md for Ojas's list as a source.
"""
from __future__ import annotations

import ast
import os
import re
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from phase2_clean_filter import normalize_surname, transliteration_stem  # noqa: E402

MAPPINGS_DIR = os.path.join(REPO_ROOT, "data", "mappings")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")

FREQ_FLOOR = 30
GENDER_HIGH, GENDER_LOW = 0.6, 0.4
CLF_ONLY_THRESHOLD = None  # disabled -- see module docstring for why (real, repeated
                            # classifier errors got through at every threshold tried)
TARGET_FIRSTNAME_PER_GENDER = 50
TARGET_SURNAME_PER_REGION = 25   # Hindu/Muslim/Christian, x4 regions ~= 100
TARGET_SIKH_SURNAME_POOLED = 100  # Sikh surnames stay pooled, not region-split
                                   # (matches Ojas's own list structure -- Sikh
                                   # surnames concentrate too heavily in Punjab
                                   # for a 4-way regional split to make sense)

REGIONS = ["North", "East", "South", "West"]
NORTHEAST_TO_EAST = {"Northeast": "East"}  # same fold as DECISIONS.md #1

NAMES_TO_CLASSIFY_PATH = os.path.join(PROCESSED_DIR, "expand_corpus_names_to_classify.txt")
EXTRA_PREDICTIONS_PATH = os.path.join(PROCESSED_DIR, "expand_corpus_predictions.csv")
OUT_PATH = os.path.join(PROCESSED_DIR, "name_bank_expanded.csv")


def load_user_lists() -> tuple[dict, dict]:
    """Returns (firstname_candidates, surname_candidates):
    firstname_candidates[(religion, gender)] = set of lowercase names
    surname_candidates[(religion, region_or_ALL)] = set of lowercase names
    """
    src_path = os.path.join(MAPPINGS_DIR, "user_provided_names_raw.py")
    with open(src_path) as f:
        tree = ast.parse(f.read())
    ns = {}
    exec(compile(tree, src_path, "exec"), ns)  # trusted, project-authored file
    first_bank, region_map = ns["FIRST_NAME_BANK"], ns["REGION_SURNAME_MAP"]

    firstnames = {}
    for key, names in first_bank.items():
        religion, gender_word = key.rsplit("_", 1)
        gender = "F" if gender_word == "Female" else "M"
        firstnames.setdefault((religion, gender), set()).update(n.lower() for n in names)

    surnames = {}
    for key, names in region_map.items():
        if "_" in key:
            religion, region = key.rsplit("_", 1)
            region = NORTHEAST_TO_EAST.get(region, region)
        else:
            religion, region = key, "ALL"
        surnames.setdefault((religion, region), set()).update(n.lower().replace("'", "") for n in names)

    return firstnames, surnames


def load_our_marker_lists() -> tuple[dict, dict]:
    firstnames = {}
    for religion, fname in [("Muslim", "muslim_firstname_markers.csv"), ("Hindu", "hindu_firstname_markers.csv")]:
        names = set(pd.read_csv(os.path.join(MAPPINGS_DIR, fname))["name"].str.lower())
        firstnames.setdefault((religion, None), set()).update(names)

    christian_fn = pd.read_csv(os.path.join(MAPPINGS_DIR, "christian_firstname_markers.csv"))
    firstnames.setdefault(("Christian", None), set()).update(christian_fn["name"].str.lower())

    sikh_fn = pd.read_csv(os.path.join(MAPPINGS_DIR, "sikh_firstname_markers.csv"))
    firstnames.setdefault(("Sikh", None), set()).update(sikh_fn["name"].str.lower())

    surnames = {}
    surnames.setdefault(("Muslim", "ALL"), set()).update(pd.read_csv(os.path.join(MAPPINGS_DIR, "muslim_surname_markers.csv"))["surname"].str.lower())
    surnames.setdefault(("Hindu", "ALL"), set()).update(pd.read_csv(os.path.join(MAPPINGS_DIR, "hindu_upper_caste_surnames.csv"))["surname"].str.lower())
    surnames.setdefault(("Sikh", "ALL"), set()).update(pd.read_csv(os.path.join(MAPPINGS_DIR, "sikh_surname_candidates.csv"))["surname"].str.lower())
    christian_sn = pd.read_csv(os.path.join(MAPPINGS_DIR, "christian_surname_markers.csv"))
    for anchor, region in [("Kerala", "South"), ("Goa", "West")]:
        names = set(christian_sn[christian_sn["anchor_state"] == anchor]["surname"].str.lower())
        surnames.setdefault(("Christian", region), set()).update(names)

    return firstnames, surnames


def all_candidate_names(fn_dicts: list[dict], sn_dicts: list[dict]) -> set[str]:
    names = set()
    for d in fn_dicts + sn_dicts:
        for s in d.values():
            names.update(s)
    return names


def cmd_prepare() -> None:
    user_fn, user_sn = load_user_lists()
    our_fn, our_sn = load_our_marker_lists()

    already_classified = set()
    for path in (os.path.join(PROCESSED_DIR, "chaturvedi_predictions.csv"),):
        if os.path.exists(path):
            already_classified.update(pd.read_csv(path)["name"].str.lower())

    universe = all_candidate_names([user_fn, our_fn], [user_sn, our_sn])
    to_classify = sorted(universe - already_classified)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(NAMES_TO_CLASSIFY_PATH, "w") as f:
        f.write("\n".join(to_classify))
    print(f"Candidate universe: {len(universe)} names ({len(universe & already_classified)} already classified)")
    print(f"Wrote {NAMES_TO_CLASSIFY_PATH} ({len(to_classify)} names needing classification)")
    print("\nNext: run under .venv-chaturvedi-lr:")
    print(f"  python src/chaturvedi_classify.py {NAMES_TO_CLASSIFY_PATH} {EXTRA_PREDICTIONS_PATH}")
    print("Then: python src/expand_corpus.py finalize")


def load_combined_clf() -> pd.DataFrame:
    parts = []
    base = os.path.join(PROCESSED_DIR, "chaturvedi_predictions.csv")
    if os.path.exists(base):
        parts.append(pd.read_csv(base)[["name", "clf_predicted_religion", "clf_max_prob"]])
    if os.path.exists(EXTRA_PREDICTIONS_PATH):
        parts.append(pd.read_csv(EXTRA_PREDICTIONS_PATH)[["name", "clf_predicted_religion", "clf_max_prob"]])
    if not parts:
        raise RuntimeError(f"No classifier predictions found. Run the 'prepare' step and the classifier first.")
    combined = pd.concat(parts, ignore_index=True)
    combined["name"] = combined["name"].str.lower()
    return combined.drop_duplicates(subset="name", keep="first")


def build_canonical_map(names_and_counts: pd.Series, protect: set[str] | None = None) -> dict:
    """names_and_counts: total count per raw spelling (already summed across
    whatever grouping matters -- e.g. nationally, or within one gender
    bucket). Returns {raw_spelling: canonical_spelling}, canonical being
    whichever spelling has the highest count within its transliteration
    stem group (Phase 2's same two deterministic rules -- see
    transliteration_stem's docstring for why this isn't generic
    edit-distance clustering).

    `protect`: raw spellings that must always map to themselves, never to
    a different stem-mate. Without this, a real hand-list candidate whose
    stem happens to collide with a bigger, unrelated real word (found by
    inspection: mann->manna, walia->wali, banerjee->banerji, masih->masiha,
    baig->baiga, vyas->vyasa, uppal->uppala, khaira->khair, and 8 more --
    18 in total across every religion, all real curated candidates,
    several with tens of thousands of real people behind them) gets
    silently renamed away in the frequency table. Every later step that
    matches candidates by their original hand-list spelling then finds no
    row left under that spelling and the name vanishes from the corpus
    with no error or warning -- caught by explicitly diffing candidate
    names against final output, not by any test. `protect` fixes this by
    only ever folding an *unprotected* spelling into the group's biggest
    member (protected or not); a protected name always keeps its own
    spelling and its own real count. See DECISIONS.md #11."""
    protect = protect or set()
    df = names_and_counts.reset_index()
    df.columns = ["name", "n"]
    df["stem"] = df["name"].map(transliteration_stem)
    df = df.sort_values("n", ascending=False)

    mapping: dict[str, str] = {}
    for _, group in df.groupby("stem", sort=False):
        names_in_group = group["name"].tolist()  # already sorted by n desc
        top = names_in_group[0]
        for n in names_in_group:
            mapping[n] = n if n in protect else top
    return mapping


def load_freq_tables(protected_surnames: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    fn = pd.read_csv(os.path.join(PROCESSED_DIR, "freq_firstnames.csv"))
    fn["first_name"] = fn["first_name"].map(normalize_surname)
    fn = fn.dropna(subset=["first_name"])
    fn = fn.groupby(["region", "first_name"])[["n_female", "n_male", "n_third_gender", "n_total"]].sum().reset_index()
    fn["prop_female"] = fn["n_female"] / fn["n_total"]

    sn = pd.read_csv(os.path.join(PROCESSED_DIR, "freq_surnames.csv"))
    sn["surname"] = sn["surname"].map(normalize_surname)
    sn = sn.dropna(subset=["surname"])
    sn = sn.groupby(["region", "surname"])["n_total"].sum().reset_index()

    # Merge transliteration spelling variants (rekhaa -> rekha, mandala ->
    # mandal) using national totals to pick one canonical spelling per
    # stem, consistently across every region -- doing this per-region
    # independently could pick a different canonical spelling in each
    # region and never merge back together. Surnames aren't gendered, so
    # this is safe to do directly (see the first-name path below for why
    # that one needs a gender-aware version of the same idea).
    # `protected_surnames` (every hand-list candidate, across every
    # religion) keeps its own spelling through this merge -- see
    # build_canonical_map's docstring for why that's needed.
    national_totals = sn.groupby("surname")["n_total"].sum()
    canonical_map = build_canonical_map(national_totals, protect=protected_surnames)
    sn["surname"] = sn["surname"].map(canonical_map)
    sn = sn.groupby(["region", "surname"])["n_total"].sum().reset_index()

    return fn, sn


def build_source_tags(name: str, religion: str, user_d: dict, our_d: dict, is_firstname: bool) -> list[str]:
    tags = []
    for key, names in user_d.items():
        if key[0] == religion and name in names:
            tags.append("ojas_list")
            break
    for key, names in our_d.items():
        if key[0] == religion and name in names:
            tags.append("marker_list")
            break
    return tags


def select_firstnames(religion: str, gender: str, user_fn: dict, our_fn: dict, fn_freq: pd.DataFrame, clf: pd.DataFrame) -> pd.DataFrame:
    hand_names = set()
    for key, names in user_fn.items():
        if key == (religion, gender):
            hand_names |= names
    for key, names in our_fn.items():
        if key[0] == religion:
            hand_names |= names  # our marker lists aren't gender-split; gender decided by real data below

    pooled = fn_freq.groupby("first_name").agg(
        n_total=("n_total", "sum"), n_female=("n_female", "sum"), n_male=("n_male", "sum")
    ).reset_index()
    pooled["prop_female"] = pooled["n_female"] / pooled["n_total"]
    pooled["gender"] = None
    pooled.loc[pooled["prop_female"] >= GENDER_HIGH, "gender"] = "F"
    pooled.loc[pooled["prop_female"] <= GENDER_LOW, "gender"] = "M"

    # Merge transliteration variants *within* the target gender only --
    # arun/aruna are a real male/female pair (see build_canonical_map's
    # caller note), so merging before splitting by gender would wrongly
    # fold a woman's name's count into a man's or vice versa. Restricting
    # the canonical map to rows already resolved to `gender` avoids that.
    same_gender = pooled[pooled["gender"] == gender]
    canonical_map = build_canonical_map(same_gender.set_index("first_name")["n_total"], protect=hand_names)
    pooled = pooled.copy()
    pooled.loc[pooled["gender"] == gender, "first_name"] = pooled.loc[pooled["gender"] == gender, "first_name"].map(canonical_map)
    pooled = pooled.groupby(["first_name", "gender"], dropna=False).agg(
        n_total=("n_total", "sum"), n_female=("n_female", "sum"), n_male=("n_male", "sum")
    ).reset_index()

    if CLF_ONLY_THRESHOLD is not None:
        clf_hits = clf[(clf["clf_predicted_religion"] == religion) & (clf["clf_max_prob"] >= CLF_ONLY_THRESHOLD)]["name"]
        candidates = hand_names | set(clf_hits)
    else:
        candidates = hand_names

    pool = pooled[pooled["first_name"].isin(candidates) & (pooled["gender"] == gender) & (pooled["n_total"] >= FREQ_FLOOR)]
    pool = pool.sort_values("n_total", ascending=False).head(TARGET_FIRSTNAME_PER_GENDER).copy()

    pool = pool.merge(clf.rename(columns={"name": "_clf_name"}), left_on="first_name", right_on="_clf_name", how="left").drop(columns=["_clf_name"])
    pool["sources"] = pool["first_name"].map(lambda n: ";".join(build_source_tags(n, religion, user_fn, our_fn, True)) or "classifier_only")
    pool["religion"], pool["region"], pool["type"] = religion, "ALL", "first"
    return pool.rename(columns={"first_name": "name", "n_total": "n"})[
        ["name", "type", "gender", "religion", "region", "n", "sources", "clf_predicted_religion", "clf_max_prob"]
    ]


def select_surnames_region(religion: str, region: str, hand_names: set, sn_freq: pd.DataFrame, clf: pd.DataFrame, target: int, user_sn: dict, our_sn: dict) -> pd.DataFrame:
    region_freq = sn_freq[sn_freq["region"] == region] if region != "ALL" else sn_freq.groupby("surname")["n_total"].sum().reset_index()
    if CLF_ONLY_THRESHOLD is not None:
        clf_hits = clf[(clf["clf_predicted_religion"] == religion) & (clf["clf_max_prob"] >= CLF_ONLY_THRESHOLD)]["name"]
        candidates = hand_names | set(clf_hits)
    else:
        candidates = hand_names

    pool = region_freq[region_freq["surname"].isin(candidates) & (region_freq["n_total"] >= FREQ_FLOOR)]
    pool = pool.sort_values("n_total", ascending=False).head(target).copy()
    pool = pool.merge(clf.rename(columns={"name": "_clf_name"}), left_on="surname", right_on="_clf_name", how="left").drop(columns=["_clf_name"])
    pool["sources"] = pool["surname"].map(lambda n: ";".join(build_source_tags(n, religion, user_sn, our_sn, False)) or "classifier_only")
    pool["religion"], pool["region"], pool["type"], pool["gender"] = religion, region, "last", None
    return pool.rename(columns={"surname": "name", "n_total": "n"})[
        ["name", "type", "gender", "religion", "region", "n", "sources", "clf_predicted_religion", "clf_max_prob"]
    ]


def cmd_finalize() -> None:
    user_fn, user_sn = load_user_lists()
    our_fn, our_sn = load_our_marker_lists()

    protected_surnames = set()
    for d in (user_sn, our_sn):
        for names in d.values():
            protected_surnames |= names
    fn_freq, sn_freq = load_freq_tables(protected_surnames=protected_surnames)
    clf = load_combined_clf()

    parts = []
    print("First names (target: {} per gender, {} per religion):".format(TARGET_FIRSTNAME_PER_GENDER, TARGET_FIRSTNAME_PER_GENDER * 2))
    for religion in ["Muslim", "Hindu", "Christian", "Sikh"]:
        for gender in ["F", "M"]:
            df = select_firstnames(religion, gender, user_fn, our_fn, fn_freq, clf)
            parts.append(df)
            print(f"  {religion:10s} {gender}: {len(df)}/{TARGET_FIRSTNAME_PER_GENDER}")

    print(f"\nSurnames (target: {TARGET_SURNAME_PER_REGION}/region for Muslim/Hindu/Christian, {TARGET_SIKH_SURNAME_POOLED} pooled for Sikh):")
    for religion in ["Muslim", "Hindu", "Christian"]:
        for region in REGIONS:
            hand_names = set()
            for key, names in user_sn.items():
                if key == (religion, region):
                    hand_names |= names
            for key, names in our_sn.items():
                if key[0] == religion and key[1] in (region, "ALL"):
                    hand_names |= names
            df = select_surnames_region(religion, region, hand_names, sn_freq, clf, TARGET_SURNAME_PER_REGION, user_sn, our_sn)
            parts.append(df)
            print(f"  {religion:10s} {region:6s}: {len(df)}/{TARGET_SURNAME_PER_REGION}")

    hand_names = set()
    for key, names in user_sn.items():
        if key[0] == "Sikh":
            hand_names |= names
    for key, names in our_sn.items():
        if key[0] == "Sikh":
            hand_names |= names
    df = select_surnames_region("Sikh", "ALL", hand_names, sn_freq, clf, TARGET_SIKH_SURNAME_POOLED, user_sn, our_sn)
    parts.append(df)
    print(f"  {'Sikh':10s} {'pooled':6s}: {len(df)}/{TARGET_SIKH_SURNAME_POOLED}")

    bank = pd.concat(parts, ignore_index=True)
    bank["human_validated"] = False
    bank.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} ({len(bank)} rows, {bank['name'].nunique()} unique names)")
    print(bank.groupby(["religion", "type"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "finalize"):
        print("Usage: python src/expand_corpus.py [prepare|finalize]", file=sys.stderr)
        sys.exit(1)
    {"prepare": cmd_prepare, "finalize": cmd_finalize}[sys.argv[1]]()
