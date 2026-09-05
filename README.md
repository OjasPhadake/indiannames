# indiannames

Gender, religion, region name bank for India, and a counterfactual ranking harness for auditing LLM hiring bias.

Full build plan: [`docs/PLAN.md`](docs/PLAN.md). Design decisions (caste handling, region mapping, validation policy): [`DECISIONS.md`](DECISIONS.md).

## Status

Phase 1 (frequency base) in progress.

## Layout

```
data/
  mappings/     state_region_mapping.csv (§3.1 / DECISIONS.md #1)
  raw/          unmodified pulls from naampy / instate
  processed/    freq_firstnames.csv, freq_surnames.csv, clean_names.csv, name_bank.csv
  validation/   hand-validation samples for the Christian/Sikh cells (DECISIONS.md #3)
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
