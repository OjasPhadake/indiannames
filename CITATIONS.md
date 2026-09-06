# Citations and data provenance

Every external paper, dataset, and tool this project draws on, and exactly what was taken from each. Kept current as the project progresses — a new source gets an entry here in the same commit that starts using it. See also [`DECISIONS.md`](DECISIONS.md) (why choices were made) and [`data/PROVENANCE.md`](data/PROVENANCE.md) (data-pull mechanics and outage workarounds) — this file is the index of *sources*; those are the record of *decisions* and *pulls*.

---

## Academic papers

### Sood, G. & Dhingra, A. (2023). "Instate: Predicting the State of Residence From Last Name." arXiv:2303.06823.
**What we took:** the underlying corpus — ~438M Indian electoral-roll voter records, 33 states, ~1.14M unique surname spellings. This is the actual source population behind every surname count in `freq_surnames.csv` / `clean_names.csv` / `name_bank.csv`. We don't cite it decoratively — every number in this repo's surname data traces back to this corpus. Full data DOI: Harvard Dataverse `10.7910/DVN/ZXMVTJ` (currently down, see `data/PROVENANCE.md`).
**Where used:** `docs/PLAN.md` §2; `src/phase1_frequency_base.py` docstring.

### Thorat, S. & Attewell, P. (2007). "The Legacy of Social Exclusion: A Correspondence Study of Job Discrimination in India." *Economic and Political Weekly*, 42(41).
**What we took:** the audit-study *method* — 4,808 fictitious job applications with names identifiable as upper-caste Hindu, Dalit, or Muslim, sent to real employers. We're following this design pattern (name-swap audit), not their specific names. Their "upper-caste Hindu" baseline is the direct model for our [`DECISIONS.md` #2](DECISIONS.md) choice to hold caste constant in the Hindu cell, and several surnames in `data/mappings/hindu_upper_caste_surnames.csv` (Sharma, Verma, etc.) reflect the same stereotype category they used, though we did not copy their specific name list (not available to us) — ours is independently hand-curated.
**Where used:** `docs/PLAN.md` §2, §3.2; `DECISIONS.md` #2, #4; `src/phase3_religion_labeling.py` docstring.

### Banerjee, A., Bertrand, M., Datta, S., & Mullainathan, S. (2009). "Labor Market Discrimination in Delhi: Evidence from a Field Experiment." *Journal of Comparative Economics*, 37(1).
**What we took:** the same method precedent as Thorat & Attewell, this time with explicit named surnames as caste markers — Sharma, Bhatia, Aggarwal (upper-caste) vs. Paswan, Manjhi, Pasi (Dalit/caste-linked). We cite their specific examples in `docs/PLAN.md` and in `DECISIONS.md`'s discussion of why "Hindu" needs the caste axis controlled (Sharma and Paswan are both Hindu but behave very differently in the audit literature — their finding, not ours).
**Where used:** `docs/PLAN.md` §2, §3.2; `DECISIONS.md` #2.

### Chaturvedi, R. & Chaturvedi, S. (2023). "It's All in the Name: A Character Based Approach to Infer Religion." arXiv:2010.14479.
**What we took:** the multiclass religion classifier (Hindu/Muslim/Christian/Sikh/Jain/Buddhist), trained on NCAER REDS data + 20k hand-annotated rural UP household heads. Two sources for this paper turned out to matter, not one:
- Repo `RochanaChaturvedi/it-is-all-in-the-name` on GitHub: **no usable weights.** Its `Multiclass.ipynb`/`religion.ipynb` notebooks only ever loaded model files from the author's private Google Drive path, and the repo's `models/` directory ships a different task's CNN plus an orphaned SVM `.sav` with no paired vectorizer.
- **The paper's official replication data**, separately hosted on Harvard Dataverse (DOI `10.7910/DVN/JOEVPN`, `its_all_in_the_name.zip`, CC0 1.0) — found by Ojas on 2026-09-06 after Dataverse recovered from its outage. This one **does** contain real, loadable `model_multiclass_lr_concat_False.sav` + vectorizer + label encoder. Used as an independent cross-check on our hand-curated marker lists (`src/chaturvedi_classify.py`, merged into `name_bank.csv` as `clf_scores`/`clf_agrees`). 83% agreement with the marker-list labels across the names both could score.
Getting the weights to load *correctly* (not just without crashing) required matching the pickle's exact scikit-learn version (1.0.2) in a dedicated Python 3.8 venv — a version-mismatched load under modern sklearn silently degenerates to predicting "Hindu" for almost everything instead of raising an error. Full debugging trail in `DECISIONS.md` #4.
**Where used:** `docs/PLAN.md` §2; `DECISIONS.md` #4; `data/PROVENANCE.md`; `src/phase3_religion_labeling.py`, `src/chaturvedi_classify.py`.

### Susewind, R. (2015). "What's in a Name? Probabilistic Inference of Religious Community from South Asian Names." *Field Methods*, 27(4). Repo: `raphael-susewind/name2community`.
**What we took:** noted as a candidate religion-labelling method (dictionary matching against a master name list), but not used — the plan flags it as low-coverage (can't classify unseen names/spelling variants) and requiring us to build our own master list first, which doesn't save work over the marker-list approach we ended up using anyway.
**Where used:** `docs/PLAN.md` §2 only; not used in code.

### (Unnamed authors, AIES 2025). "Invisible Filters" [working title as given in the plan — exact citation not yet verified against the published proceedings].
**What we took:** the closest prior LLM-specific audit work — varies names by gender, caste, region in Indian interview transcripts, finds no significant region/caste effect and marginal gender effects failing Tukey post-hoc (50 transcripts × 8 conditions = 400 points, likely underpowered). This is the null result our own power calculation (plan §3.4, not yet run) needs to be powered against.
**Where used:** `docs/PLAN.md` §2, §3.4, §7.
**TODO:** get the full/formal citation before this goes in any paper — currently only have the working title and venue from the plan document, not a verified author list or DOI.

### Vahini, K. et al. arXiv:2209.03089. "Decoding Demographic un-fairness from Indian Names."
**What we took:** nothing — logged as a deliberate non-source. Covers gender + caste over 7.63M unique names but has no religion axis, and the authors release only their scraping *code*, not the name datasets, as a stated privacy policy. The plan calls this "a known dead end" and instructs not to budget time chasing it; we didn't. Their code-not-data release policy is itself cited as the precedent behind our own release plan (§7 of `docs/PLAN.md`: release the name bank and code, not raw voter-record linkage).
**Where used:** `docs/PLAN.md` §2, §7.

---

## Software / tools

### `appeler/instate` — surname → state prediction
**What we took:** `instate_unique_ln_state_prop_v2.parquet` (1,915,898 surnames × 34 states, real electoral-roll counts). **Not** taken via `pip install instate` (installs 0.1.7, whose hardcoded download URL now 404s after the org's v3.0.0 rewrite). First pulled from the `v2.0.0` GitHub tag as a stopgap; switched 2026-09-05 to `https://huggingface.co/gojiberries/instate/resolve/main/instate_unique_ln_state_prop_v2.parquet` — the maintainers' own current, pinned-commit distribution point (found by the project owner, verified byte-identical to the tag version before switching). Full mechanics in `data/PROVENANCE.md`.
**License:** MIT (per the repo).
**Where used:** `src/phase1_frequency_base.py`, `src/phase3_religion_labeling.py` (Christian cell, direct read of the raw cache).

### `gojiberries/instate`, `gojiberries/naampy`, `gojiberries/pranaam` — Hugging Face, the `appeler` org's current model/data hosting
**What we took:** the instate surname table above, from `gojiberries/instate`.
**What we checked and didn't take:** `gojiberries/naampy`'s bulk file (`naampy_v2_1k_global_binary.parquet`) — inspected directly, it's a national-only aggregate (40,581 names, global male/female counts, no state column), so it can't substitute for the region-level first-name data Phase 1 needs. `gojiberries/pranaam`'s model weights (real safetensors files, English + Hindi) — exist, but the released `pranaam` package doesn't yet contain the code to load them (that's `pranaam/model_v3.py` on the GitHub `main` branch, unreleased and not wired into a public function).
**Where used:** `src/phase1_frequency_base.py`; `data/PROVENANCE.md`.

### `appeler/naampy` — first name → state/gender
**What we tried to take:** first-name × state × gender counts (v2 dataset, ≥100 occurrences/name, 30 states). **Blocked**, not obtained yet — naampy downloads this from Harvard Dataverse at runtime and every Dataverse endpoint has been 504ing since 2026-09-05. No alternative source found after checking: Internet Archive's "India Names Dataset" (only ever contained Andhra Pradesh, despite the title — see below), several Kaggle datasets (no region breakdown, undocumented provenance), census.name (paid, social-media-scraped, no region breakdown), and `gojiberries/naampy` on Hugging Face (real, but national-only, no state column — see below). Tracked by an automated recovery routine (`trig_01AiyrThAMwq9BhaTmjebsow`) that retries every 3 hours and pulls this the moment Dataverse recovers.
**Where used:** `src/phase1_frequency_base.py`; blocker documented in `data/PROVENANCE.md`.

### `appeler/pranaam` — Muslim/non-Muslim classifier
**What we tried to take:** binary religion prediction (`pred_rel()`) to replace the hand-curated Muslim marker list with an actual classifier cross-check. **Blocked** — installed fine (0.9.0), but its model weights download from Harvard Dataverse at call time (`https://dataverse.harvard.edu/api/access/datafile/6286241`), same outage as naampy. Not used; `src/phase3_religion_labeling.py` falls back to a hand-curated marker list instead, documented as a stopgap in `DECISIONS.md` #4.
**Where used:** attempted in `src/phase3_religion_labeling.py` investigation, not currently called by any script.

---

## Sources checked and explicitly not used

Kept here so we don't re-spend time re-evaluating these later.

| Source | Why not used |
|---|---|
| Internet Archive, "India Names Dataset" (uploader `anandology-2`, 2015, CC-BY 4.0) | Titled/described as covering all of India, but its own user reviews confirm the archive only ever contained one state's file (Andhra Pradesh). Predates naampy; same underlying idea, incomplete in practice. |
| `census.name` Indian Name Database | Paid (€55). Their own methodology page states the data mixes "government data" with **scraped social media profiles** — no state/region breakdown, opaque and ethically questionable sourcing. Not appropriate for a research pipeline. |
| Kaggle Indian-names/gender datasets (several: `pritsheta/indian-male-and-female-name-dataset`, `dex0510/indian-gendername-dataset`, `shubhamuttam/indian-names-by-gender`, others) | None document a state/region breakdown or a verifiable sourcing methodology; several are plausibly re-derivations of naampy/instate's own output, which would make citing them circular. |
| `RochanaChaturvedi/it-is-all-in-the-name` GitHub repo specifically (as opposed to the paper's Dataverse replication data, which we do use — see academic papers section above) | No published weights in the repo itself, not a temporary outage. |
| `gojiberries/pranaam`'s Hugging Face weights | Real files, but no released version of the `pranaam` package can load them yet. |
| Rebuilding naampy from raw electoral-roll PDFs ourselves (`in-rolls/electoral_rolls`, `in-rolls/parse_searchable_rolls`) | Technically possible (this is naampy's own upstream raw material) but is redoing naampy's entire OCR/parsing pipeline from scratch — realistically days of work, not a shortcut around the Dataverse outage (which recovered 2026-09-06 anyway). |
| Zenodo restricted-access raw electoral-roll archive (mentioned in search results as a companion to the Dataverse copy) | Access-gated, requires an academic access request — moot now that Dataverse recovered, but worth knowing about for a paper's data-availability statement regardless. |
| SVM variant of Chaturvedi's classifier (`model_multiclass_svm_concat_False.sav`) | Loadable (under a matching sklearn 0.22.2.post1 environment), but less accurate than the LR variant in our own spot-check, and `LinearSVC.decision_function()` gives a margin, not the probability the plan's threshold approach wants. LR used instead. |

---

## Not cited: hand-curated content

The marker-surname lists in `data/mappings/` (`muslim_surname_markers.csv`, `hindu_upper_caste_surnames.csv`, `christian_surname_markers.csv`, `sikh_surname_candidates.csv`) were **typed by hand from general knowledge of Indian surname–religion/caste associations, not drawn from any paper or dataset**. They follow the *method* of Thorat & Attewell / Banerjee et al. (hand-picked stereotype surnames) but are not their specific lists, which aren't published in a form we could reuse. This is flagged prominently, not buried, because it's the least rigorous part of the pipeline right now — see `DECISIONS.md` #4 for the plan to replace/cross-check it once pranaam is reachable again.

### Ojas's `FIRST_NAME_BANK` / `REGION_SURNAME_MAP` (2026-09-06)
**What it is:** a hand-typed reference list of ~40-90 names per religion/gender/region cell, provided by Ojas for his own downstream tool, kept verbatim at `data/mappings/user_provided_names_raw.py`. Same nature as this project's own marker lists above — real names, but hand-typed from general knowledge, not derived from data.
**What we took:** used as one of three merge inputs (alongside our marker lists and the Chaturvedi classifier) for `src/expand_corpus.py`'s expanded ~100/religion corpus (`data/processed/name_bank_expanded.csv`) — see `DECISIONS.md` #5 for the full merge methodology, including why the classifier ended up not being used as a candidate source despite being tried. Names from this list only make it into the final output if they also clear a real electoral-roll frequency floor; nothing is included on the list's say-so alone.
**Where used:** `src/expand_corpus.py`; `DECISIONS.md` #5.
