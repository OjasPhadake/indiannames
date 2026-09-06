# indiannames

Gender, religion, region name bank for India, and a counterfactual ranking harness for auditing LLM hiring bias.

Full build plan: [`docs/PLAN.md`](docs/PLAN.md). Design decisions (caste handling, region mapping, validation policy, classifier availability): [`DECISIONS.md`](DECISIONS.md). Every paper/dataset/tool used and what was taken from each: [`CITATIONS.md`](CITATIONS.md). Data-pull mechanics and outage workarounds: [`data/PROVENANCE.md`](data/PROVENANCE.md).

## Status

Phases 1-3 done for both first names and surnames (`data/processed/name_bank.csv`, 241 rows across Muslim/Hindu/Christian/Sikh). Religion labels are primarily hand-curated-marker-based, cross-checked against Chaturvedi & Chaturvedi's multiclass classifier (83% agreement) — see `DECISIONS.md` #4 for the full story, including why that classifier needs its own separate Python environment.

## Layout

```
data/
  mappings/     state_region_mapping.csv, *_surname_markers.csv / *_firstname_markers.csv (religion marker lists)
  raw/          unmodified pulls from naampy / instate
  processed/    freq_firstnames.csv, freq_surnames.csv, clean_names.csv, chaturvedi_predictions.csv, name_bank.csv
  validation/   near_duplicates_for_review.csv, hand-validation samples (DECISIONS.md #3)
src/            build scripts, one per phase
docs/           PLAN.md
```

## Setup

Main pipeline (phases 1, 2, and the marker-list part of phase 3):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install 'setuptools<81'  # naampy needs pkg_resources, which setuptools>=81 removed
pip install -r requirements.txt
```

The Chaturvedi classifier cross-check needs its own environment — its model pickles were made with `scikit-learn==1.0.2`, which is incompatible with the modern sklearn the main venv uses (see `DECISIONS.md` #4 for what goes wrong if you skip this and try to load it in the main venv anyway):

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update && sudo apt install -y python3.8 python3.8-venv python3.8-dev
python3.8 -m venv .venv-chaturvedi-lr
source .venv-chaturvedi-lr/bin/activate
pip install numpy==1.21.6 scipy==1.7.3 scikit-learn==1.0.2 pandas
```

Model files (not in this repo — six files, ~600MB, source: Harvard Dataverse DOI `10.7910/DVN/JOEVPN`, CC0) go in `~/.chaturvedi/models/`; see `data/PROVENANCE.md` for exact filenames and the download.

## Build phases

1. Frequency base (region × gender) — `src/phase1_frequency_base.py`
2. Clean and filter
3. Religion labelling
4. Name-familiarity control
5. Resume generation
6. Ranking harness
7. Analysis

See `docs/PLAN.md` §4 for the detail behind each phase.
