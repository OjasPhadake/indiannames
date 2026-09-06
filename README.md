# Counterfactual Resume Audit of LLM Hiring Bias — India

A religion × gender × region name bank for India, built from real 2017 electoral-roll frequency data rather than invented or generically-sourced names, plus the evaluation harness for a counterfactual resume-ranking audit of LLM hiring bias.

**Author:** Ojas Phadake · **Status:** name bank complete (Phases 1–3), harness (Phases 4–7) in progress · **License:** see [Data provenance & licensing](#data-provenance--licensing)

---

## Why this exists

Auditing an LLM ranker for hiring bias by name requires a name bank where religion, gender, and region are each independently real and verifiable — not names an author recognizes as "sounding" a certain way. Every name in this repository is grounded in the 2017 Indian electoral rolls (~438M records, Sood & Dhingra 2023) rather than hand-guessed, and every design decision — caste handling, region folding, classifier reliability, what counts as "resolved" when two communities share a spelling — is recorded with its reasoning in [`DECISIONS.md`](DECISIONS.md), not left implicit in code.

## Results at a glance

Two name banks are produced, at different points on the precision/recall trade-off:

| | `name_bank.csv` | `name_bank_expanded.csv` |
|---|---|---|
| **Purpose** | Primary, tightly-validated bank (`docs/PLAN.md`'s original design) | Higher-volume bank for studies needing more names per cell |
| **Rows** | 241 | 707 (608 unique names) |
| **Gender purity gate** | ≥95% / ≤5% | ≥60% / ≤40% (ambiguous names excluded either way) |
| **Frequency floor** | 500 real people/region | 30 real people |
| **Target size** | ~25 first names + ~25 surnames / religion | ~100 first names + ~100 surnames / religion |

`name_bank_expanded.csv` breakdown (first names by gender, surnames by region; Sikh surnames are pooled nationally rather than region-split — see [`DECISIONS.md` #5](DECISIONS.md)):

| Religion | First (F/M) | Surnames (N/E/S/W or pooled) | Min. real count |
|---|---|---|---|
| Hindu | 50 / 50 | 25 / 25 / 25 / 25 | 579 |
| Muslim | 47 / 50 | 25 / 25 / 25 / 25 | 41 |
| Christian | 47 / 50 | 13 / 25 / 25 / 25 | 43 |
| Sikh | 21 / 43 | 61 (pooled) | 106 |

Every count is a real person-count from the source electoral rolls, not a model score. Sikh and Christian cells are smaller in absolute terms than Hindu ones by design, not by omission — see [Known limitations](#known-limitations).

## Repository layout

```
docs/
  PLAN.md                         Original 7-phase research plan
data/
  mappings/
    state_region_mapping.csv       State -> {North,East,South,West} fold
    *_firstname_markers.csv        Hand-curated per-religion first-name markers (this project's own)
    *_surname_markers.csv          Hand-curated per-religion surname markers (this project's own)
    hindu_upper_caste_surnames.csv Caste-constant Hindu baseline (Thorat & Attewell / Banerjee et al. precedent)
    chaturvedi_train_class_prior.csv  Real training-class balance for the Chaturvedi classifier
    user_provided_names_raw.py     Hand-typed FIRST_NAME_BANK / REGION_SURNAME_MAP — canonical source list,
                                    QA'd across five rounds (DECISIONS.md #7-#11)
  processed/
    freq_firstnames.csv, freq_surnames.csv   Phase 1 output — real frequency by region/gender
    clean_names.csv                          Phase 2 output — cleaned, deduplicated, floor=500
    chaturvedi_predictions.csv               Chaturvedi classifier run on clean_names.csv
    name_bank.csv                            Phase 3 output — the primary, tightly-validated bank
    expand_corpus_names_to_classify.txt,
    expand_corpus_predictions.csv            Two-pass classifier hand-off for the expanded bank
    name_bank_expanded.csv                   The expanded, higher-volume bank
    name_bank_expanded_dict_format.py        Same data as a Python dict, matching the original hand-list shape
    broad_pool_*.csv, testcases_*.csv,
    soft_score_*.csv,
    classifier_only_vs_handlist_overlap.csv  Outputs of three rejected alternative methods, kept as documented
                                              negative results (DECISIONS.md #6, #9)
  validation/
    near_duplicates_for_review.csv           Edit-distance-1 pairs flagged, not auto-merged (DECISIONS.md #7)
  PROVENANCE.md                   Data-pull mechanics, upstream outages, and exact URLs used
src/
  phase1_frequency_base.py        Pull real region x gender frequency tables from naampy / instate
  phase2_clean_filter.py          Clean, dedupe transliteration variants, apply the floor
  phase3_religion_labeling.py     Label by hand-curated markers, cross-check against the classifier
  chaturvedi_classify.py          Runs the Chaturvedi & Chaturvedi multiclass religion classifier
                                   (needs its own Python 3.8 / sklearn 0.22.2.post1 or 1.0.2 environment)
  expand_corpus.py                Two-pass build of the expanded bank: merges the hand lists + this
                                   project's markers + the classifier, real-frequency-ranked
  generate_dict_export.py         Re-exports name_bank_expanded.csv as the FIRST_NAME_BANK /
                                   REGION_SURNAME_MAP dict shape
  prior_correct.py, soft_score.py,
  classifier_only_comparison.py   Three alternative "make it more classifier-driven" approaches,
                                   tested and rejected — kept for the documented negative result
DECISIONS.md                      Every design decision and QA round, with reasoning (11 sections)
CITATIONS.md                      Every paper/dataset/tool used and exactly what was taken from each
```

## Methodology summary

1. **Frequency base** (Phase 1) — real first-name and surname counts by state, folded to four regions (North/East/South/West), from the 2017 electoral rolls via `naampy` (first names) and `instate` (surnames).
2. **Clean & filter** (Phase 2) — drop relational/honorific artifacts, collapse only two explicit, deterministic transliteration-spelling rules (never generic edit-distance clustering — see the rationale in `src/phase2_clean_filter.py`'s docstring, which documents a specific dangerous merge caught and reverted), apply a frequency floor.
3. **Religion labelling** (Phase 3) — primary labels from hand-curated marker lists (the same method used in Thorat & Attewell 2007 and Banerjee et al. 2009), cross-checked against Chaturvedi & Chaturvedi's independent multiclass classifier: 83% agreement (161/194 `name_bank.csv` rows the classifier returned a verdict for).
4. **Corpus expansion** (`expand_corpus.py`) — for studies needing more names per cell, merges three candidate sources (this project's own hand list, the user-provided hand list, and the classifier as a cross-check only, never as a sole source of inclusion — see [`DECISIONS.md` #6](DECISIONS.md) for why classifier-only inclusion was tested and rejected), still ranked purely by real regional/religion-specific frequency.
5. **Five rounds of QA** ([`DECISIONS.md` #7–#11](DECISIONS.md)) — gender-purity and region-mismatch checks against real data, cross-religion duplicate detection and resolution, frequency-floor sensitivity analysis, and a pipeline bug fix that had been silently dropping real, hand-listed candidates (`Banerjee`, `Baig`, `Vyas`, `Mann`, and 14 others) since the first `expand_corpus.py` run — see `DECISIONS.md` #11 for the root cause and fix.

Every classification is either a real electoral-roll frequency measurement or a human decision with its reasoning recorded — nothing in the final bank rests on an unexamined model output. See `DECISIONS.md` for the full reasoning behind every judgment call, including the ones this project's own author changed after review.

## Known limitations

- **Religion labels are not directly observed.** The electoral rolls carry no religion field; labels come from hand-curated markers and an independent classifier, not ground truth. Every ambiguous or disputed case is logged in `DECISIONS.md`, but residual misclassification is possible for names not yet checked.
- **The classifier (Chaturvedi & Chaturvedi) has known, specific biases**, not just noise: it defaults to "Hindu" under uncertainty, and independently mislabels several confirmed-Muslim names as "Sikh" at high confidence. It is used only as a cross-check, never as a sole basis for inclusion (`DECISIONS.md` #4, #6, #9).
- **Sikh and Christian cells are population-proportionally smaller**, not lower-quality. India's 2011 census religion shares (~80% Hindu, ~14% Muslim, ~2.3% Christian, ~1.7% Sikh) mean a minority religion's most common names still carry far smaller absolute real-person counts than an equivalently-ranked Hindu name. A flat minimum-count floor was tested and rejected for exactly this reason (`DECISIONS.md` #10).
- **Caste is held constant** within the Hindu cell (upper-caste baseline, following Thorat & Attewell 2007 / Banerjee et al. 2009) — caste-based discrimination is a separate axis this study does not vary.
- **Single-year, single-source frequency snapshot** (2017 electoral rolls) — name popularity shifts over time and this bank reflects one cross-section.
- **Northeast and central Hindi-belt states are folded** into East and North respectively for regional aggregation (`DECISIONS.md` #1) — a coarser regional signal than state-level would give.

## Setup

Main pipeline (Phases 1–2, the marker-list part of Phase 3, and `expand_corpus.py`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install 'setuptools<81'  # naampy needs pkg_resources, which setuptools>=81 removed
pip install -r requirements.txt
```

The Chaturvedi classifier cross-check needs its own environment — its model pickles were made with `scikit-learn==0.22.2.post1` / `1.0.2`, incompatible with the modern sklearn the main venv uses. Loading them under the wrong version does not error; it silently degenerates to predicting the majority class (see `DECISIONS.md` #4):

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update && sudo apt install -y python3.8 python3.8-venv python3.8-dev
python3.8 -m venv .venv-chaturvedi-lr
source .venv-chaturvedi-lr/bin/activate
pip install numpy==1.21.6 scipy==1.7.3 scikit-learn==1.0.2 pandas
```

Model files (not in this repo — six files, ~600MB, source: Harvard Dataverse DOI `10.7910/DVN/JOEVPN`, CC0) go in `~/.chaturvedi/models/`; see [`data/PROVENANCE.md`](data/PROVENANCE.md) for exact filenames and download links.

## Reproducing the pipeline

**Primary bank** (`name_bank.csv`):

```bash
source .venv/bin/activate
python src/phase1_frequency_base.py
python src/phase2_clean_filter.py
source .venv-chaturvedi-lr/bin/activate
python src/chaturvedi_classify.py data/processed/clean_names.csv data/processed/chaturvedi_predictions.csv
source .venv/bin/activate
python src/phase3_religion_labeling.py
```

**Expanded bank** (`name_bank_expanded.csv`):

```bash
source .venv/bin/activate
python src/expand_corpus.py prepare
source .venv-chaturvedi-lr/bin/activate
python src/chaturvedi_classify.py data/processed/expand_corpus_names_to_classify.txt data/processed/expand_corpus_predictions.csv
source .venv/bin/activate
python src/expand_corpus.py finalize
python src/generate_dict_export.py   # optional -- re-exports as the FIRST_NAME_BANK / REGION_SURNAME_MAP dict shape
```

## Roadmap

Phases 1–3 (this name bank) are complete. Remaining, per [`docs/PLAN.md`](docs/PLAN.md) §4:

4. Name-familiarity control
5. Resume generation
6. Ranking harness (counterfactual name-swap, average rank change across LLM rankers)
7. Analysis

## Data provenance & licensing

- Surname frequencies: Sood, G. & Dhingra, A. (2023), *Instate*, arXiv:2303.06823 — 2017 Indian electoral rolls, re-hosted at `gojiberries/instate` (Hugging Face) after the original GitHub distribution point was superseded; verified byte-identical before switching.
- First-name frequencies: `naampy` (`appeler/naampy`), same electoral-roll source.
- Religion classifier: Chaturvedi & Chaturvedi, "It's All in the Name," official replication weights via Harvard Dataverse DOI `10.7910/DVN/JOEVPN` (CC0).
- Full source list and exactly what was taken from each: [`CITATIONS.md`](CITATIONS.md). Data-pull mechanics and outage workarounds: [`data/PROVENANCE.md`](data/PROVENANCE.md).

This repository's own code and hand-curated marker lists are original work by the author. Consult the upstream sources' own licenses (linked in `CITATIONS.md`) before redistributing derived frequency data.
