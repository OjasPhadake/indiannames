# indiannames

Gender, religion, region name bank for India, and a counterfactual ranking harness for auditing LLM hiring bias.

Full build plan: [`docs/PLAN.md`](docs/PLAN.md). Design decisions (caste handling, region mapping, validation policy, classifier availability): [`DECISIONS.md`](DECISIONS.md). Every paper/dataset/tool used and what was taken from each: [`CITATIONS.md`](CITATIONS.md). Data-pull mechanics and outage workarounds: [`data/PROVENANCE.md`](data/PROVENANCE.md).

## Status

Phases 1-3 done for surnames (`data/processed/name_bank.csv`). First names blocked on a Harvard Dataverse outage (tracked by an automated recovery routine); Phase 3's religion labels are hand-curated-marker-based rather than classifier-verified, since both classifiers the plan named turned out unavailable — see `DECISIONS.md` #4.

## Layout

```
data/
  mappings/     state_region_mapping.csv, *_surname_markers.csv (religion marker lists), sikh_surname_candidates.csv
  raw/          unmodified pulls from naampy / instate
  processed/    freq_firstnames.csv, freq_surnames.csv, clean_names.csv, name_bank.csv
  validation/   near_duplicates_for_review.csv, hand-validation samples (DECISIONS.md #3)
src/            build scripts, one per phase
docs/           PLAN.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build phases

1. Frequency base (region × gender) — `src/phase1_frequency_base.py`
2. Clean and filter
3. Religion labelling
4. Name-familiarity control
5. Resume generation
6. Ranking harness
7. Analysis

See `docs/PLAN.md` §4 for the detail behind each phase.
