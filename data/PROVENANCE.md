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

**Next step:** re-run `python src/phase1_frequency_base.py` once Dataverse is reachable again (`curl -o /dev/null -w '%{http_code}\n' https://dataverse.harvard.edu/` should return 200, not 504/502).
