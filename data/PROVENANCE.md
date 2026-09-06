# Data provenance

Both upstream packages named in `docs/PLAN.md` §2 turned out to have live-download problems when this project actually started pulling data (2026-09-05). Recorded here so the workarounds are auditable and revisited if upstream fixes things.

## Surnames (`freq_surnames.csv`)

**Source:** `instate_unique_ln_state_prop_v2.parquet` — 1,915,898 unique surnames x 34 states/UTs, columns are each surname's record-weighted share of its `total_n` found in that state, from the 2017 Indian electoral rolls (Sood & Dhingra, arXiv 2303.06823).

**Actual URL used (updated 2026-09-05, see history below):**
`https://huggingface.co/gojiberries/instate/resolve/main/instate_unique_ln_state_prop_v2.parquet`

**Why not the pip package:** `pip install instate` currently installs `0.1.7` (the latest release on PyPI), whose `load_instate_data()` hardcodes a download URL to `https://github.com/appeler/instate/raw/main/data/instate_unique_ln_state_prop_v1.csv.gz`. That file no longer exists on the `main` branch — the upstream repo went through a rewrite (`v3.0.0`, tagged 2026-08-19) that moved to a Hugging-Face-hosted neural model with a single-surname "abstain" API (see their `CHANGELOG.md` and `MODEL_CARD.md`) and dropped the bulk downloadable table from the live repo. The live URL now 404s.

**Source history:** first pulled from the `v2.0.0` git tag on GitHub (a snapshot of the same table, still legitimately public but effectively an archaeological workaround). Switched to the Hugging Face URL above the same day after the project owner found `gojiberries/instate`'s model card, which states the repo is served "at an immutable commit so a released package cannot silently change models" — the maintainers' own current, intentionally-versioned distribution point, on infrastructure independent of the Dataverse outage below. **Verified byte-identical** to the old git-tag source before switching (same shape 1,915,898 × 36, same `total_n` sum 699,893,539, same values in every row checked) — this is a provenance upgrade, not a data change.

**Caveat:** `total_n` and the per-state shares come from this snapshot of the electoral-roll corpus; if a future `instate` release reprocesses the underlying rolls differently, this table won't reflect that. Revisit if this project is ever revived after a long gap.

**Checked and rejected as a first-names fix:** `gojiberries/naampy` also exists on Hugging Face, but its bulk file (`naampy_v2_1k_global_binary.parquet`) is a **national-only aggregate** — 40,581 first names with just global male/female counts, no state column. The org appears to have deliberately dropped the regional breakdown for first names in this new architecture (unlike surnames, where the full state-level table is still shipped). Doesn't solve the first-names blocker below. Logged in `CITATIONS.md`.

**Checked and rejected as a pranaam fix:** `gojiberries/pranaam` also has real model weights on HF (safetensors, English + Hindi) — but the currently released pip package (0.9.0, same as the newest GitHub tag) doesn't know how to load them. The loading code for that (`pranaam/model_v3.py` on the `main` branch) is low-level, unreleased, and not wired into any public function yet. pranaam is mid-migration, same as instate was, just not far enough along. Worth rechecking in a few weeks.

## First names (`freq_firstnames.csv`)

**Source:** naampy's `v2` dataset (30 states, first names with ≥100 occurrences), `state, birth_year, first_name, n_female, n_male, n_third_gender` — pulled live from Harvard Dataverse (`10.7910/DVN/ZXMVTJ`) by `naampy.in_rolls_fn.InRollsFnData.load_naampy_data()`. naampy never bundles this table itself; there's no packaged or tagged-release fallback the way there was for `instate`.

**Status as of 2026-09-05:** every Dataverse endpoint tried — the specific datafile URLs (`.../access/datafile/496...`), the dataset metadata API, and the Dataverse homepage itself — returned HTTP 504. This looks like a Harvard Dataverse service outage, not a broken or moved link. `src/phase1_frequency_base.py` catches this, writes the surname table anyway, and skips first-name extraction with a clear message rather than fabricating placeholder data.

**Recovered 2026-09-06.** Ojas found Dataverse loading again by hand and reported it; re-ran `phase1_frequency_base.py` and it pulled successfully. Two things came up in that pull, both fixed the same day:

1. **OOM on first attempt.** `build_firstname_table` loaded the full 23.8M-row `naampy_v2.csv.gz` into memory and then immediately re-saved a full copy through pandas before aggregating — held two copies of a large frame at once, killed (exit 137) on this 8GB box. Same failure shape as the earlier surname-melt OOM in `build_surname_region_table`. Fixed by chunked aggregation (`pd.read_csv(..., chunksize=2_000_000)`, incrementally summed) and replacing the redundant pandas round-trip with a plain `shutil.copy` of the already-downloaded cache file.
2. **10 states silently dropped from the region rollup.** naampy's `state` column uses short-form/abbreviated names (`andhra`, `up`, `mp`, `jk`, `himachal`, `andaman`, `arunachal`, `dadra`, `daman`, and a misspelled `maharastra` — missing the middle "h") that didn't match `state_region_mapping.csv`'s full official names, so those 10 states' first-name records were silently excluded from every region total until caught. Added as explicit aliased rows in the mapping file (each documented inline as a naampy-specific short form, not a duplicate/typo of ours).

**Known real gap, not a bug:** naampy's `v2` dataset covers only 31 states/UTs total, and **West Bengal, Tamil Nadu, and Telangana are absent entirely** — no short-form alias exists for them because naampy simply doesn't include them in this dataset variant (naampy's own CLI docs describe `v2_native`/`v2_en` as separate variants for some of this coverage, not merged into `v2`). This means `freq_firstnames.csv`'s South and East region totals are undercounts relative to `freq_surnames.csv`'s (which does have full 34-state instate coverage) — worth flagging in any paper that uses both tables side by side, and worth trying `v2_native`/`v2_en` later if that coverage gap matters for the actual study design.

**Transliteration quirk found in the Punjab rows** while building the Sikh cell (Phase 3): this dataset's Gurmukhi-to-Roman transliteration preserves the script's implicit vowels rather than dropping them the way standard Romanization does. "Harpreet" appears in this data as `harapreeth`, "Gurmeet" as `gurameeth`, "Kuldeep" as `kuladeep`, etc. A first attempt at the Sikh given-name marker list using standard spellings matched zero names in the real data; rebuilt against the actual spellings found by inspecting Punjab's top first names directly. See `data/mappings/sikh_firstname_markers.csv` for the full corrected list with a `standard_spelling` reference column.

## Religion classifier (`chaturvedi_predictions.csv`)

**Source:** Chaturvedi & Chaturvedi's official replication data for "It's All in the Name: A Character Based Approach to Infer Religion," Harvard Dataverse, DOI `10.7910/DVN/JOEVPN`, file `its_all_in_the_name.zip` (1.4GB, CC0 1.0 license). Found by Ojas on 2026-09-06 after Dataverse recovered — the GitHub repo for this paper (cited in `docs/PLAN.md` §2) ships no usable weights at all (see `DECISIONS.md` #4), but the paper's own official replication package, hosted separately, does.

**What's actually in it (relevant subset — the full zip also contains an unrelated USA race-classification pipeline, North Carolina voter data, etc., all ignored):**
- `models/model_multiclass_lr_concat_False.sav` + `models/vectorizer_multiclass_lr_concat_False.sav` — LogisticRegression + its fitted TF-IDF vectorizer, single-name (not parent-concatenated) input. **This is the pair we use.**
- `models/model_multiclass_svm_concat_False.sav` + matching vectorizer — an SVM alternative, tried first, less accurate in our spot-check, not used.
- `models/non_neural_label_encoding_multiclass.pkl` — the label encoder (`Hindu, Muslim, Sikh, Christian, Jain, Buddhist`).
- `data/REDS_train/val/test_multiclass.csv` — the actual train/val/test splits, though the `name`/`parent` fields are anonymized in this public release (privacy protection on the underlying NCAER REDS survey data), so we can't validate accuracy against real names in the held-out test set ourselves.

**Version trap, found by testing, not assumed:** the SVM pickle was made with `scikit-learn==0.22.2.post1`; the LR pickle with `1.0.2`. Loading either under this project's modern sklearn (1.7.2) does not raise an error — it silently produces a broken TF-IDF transform, and the classifier degenerates to predicting the majority class ("Hindu") for nearly every input, including unambiguous names like Khan and Ansari. Fixed with a dedicated venv matching the LR pickle's exact version:

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update && sudo apt install -y python3.8 python3.8-venv python3.8-dev
python3.8 -m venv .venv-chaturvedi-lr
source .venv-chaturvedi-lr/bin/activate
pip install numpy==1.21.6 scipy==1.7.3 scikit-learn==1.0.2 pandas
```

Model files cached locally at `~/.chaturvedi/models/` (not committed to this repo — six files, ~600MB combined, and the source is only distributed as one non-splittable 1.4GB zip, so there's no lightweight re-fetch path; anyone reproducing this needs to download the full zip once). Run `src/chaturvedi_classify.py` under `.venv-chaturvedi-lr` to regenerate `data/processed/chaturvedi_predictions.csv`; `src/phase3_religion_labeling.py` (run under the normal project venv) merges it in automatically if present.

**Confirmed-real model weaknesses** (reproduced under the correctly-matched environment, so these are genuine limitations, not version artifacts): `fernandes` → Hindu instead of Christian; `ayesha` is a near-coinflip between Muslim and Christian (0.509 probability); most Sikh given names in this project's Punjab data predict Hindu, likely because the classifier's own training data never saw this dataset's implicit-vowel transliteration convention. Full disagreement list logged in DECISIONS.md #4.
