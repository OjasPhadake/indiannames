# Design decisions

Resolves the open items in [`docs/PLAN.md` §5](docs/PLAN.md#5-open-decisions-blocking-phase-1). Dated so later revisions are traceable.

---

## 1. State → region mapping (2026-09-05)

**Decision:** four buckets — North, South, East, West — per `docs/PLAN.md` §3.1. Full mapping lives in [`data/mappings/state_region_mapping.csv`](data/mappings/state_region_mapping.csv), not inline in any script.

Two groups don't have an obvious home in a 4-way scheme and were assigned as follows:

- **Central Hindi-belt states (Madhya Pradesh, Chhattisgarh) → North.** Grouped with North on linguistic/cultural grounds (Hindi-belt), not geography. This is the more contestable of the two calls — Chhattisgarhi in particular sits in the eastern Indo-Aryan zone. Revisit if the Hindu-only regional sub-study (§3.1) shows these two states behaving like East rather than North in the frequency tables.
- **North-East states (Assam, Arunachal Pradesh, Manipur, Meghalaya, Mizoram, Nagaland, Sikkim, Tripura) → East.** This is a coarsening, not a claim that the North-East is linguistically or ethnically part of "East India" — it isn't. It's folded in because the region factor only has four levels and a 5th "North-East" bucket would have too few names per cell once crossed with gender (§3.3's 25/cell floor). **This bucketing applies only to the region factor in the nested Hindu-only sub-study.** It does **not** apply to Phase 3 step 4 (Christian cell construction), which pulls North-East surnames specifically regardless of this region label — the two are independent uses of "region."

**Caveat carried forward:** because the region factor is Hindu-only (§3.1), this mapping only ever gets exercised on Hindu-majority states in practice. The North-East and Central assignments matter more for documentation completeness than for actual cell populations.

## 2. Caste inside the Hindu cell (2026-09-05)

**Decision: held constant.** The Hindu cell uses upper-caste surnames only, matching the Thorat & Attewell (2007) baseline (Sharma, Verma, Gupta, Bhalla, Kumar-type surnames), rather than opening caste as a fourth crossed axis.

**Why:** keeps the primary design at religion (4) × gender (2) = 8 cells as specified in §3.1, rather than doubling the Hindu-cell name count and adding a factor to the mixed-effects model. Caste-linked audit effects (Banerjee et al. 2009) are a separate, well-studied axis — worth a follow-up study, not a confound to fold into this one.

**Consequence:** `name_bank.csv` rows for `religion=Hindu` should be filtered/tagged to upper-caste surnames during Phase 3 construction (not left as a raw draw from the general Hindu-classified population), and this filter needs to be documented in the row's `source`/`clf_scores` provenance fields so it's auditable later.

## 3. Hand validation of Christian / Sikh cells (2026-09-05)

**Decision: deferred**, not skipped. Per Phase 3 step 5 and the risk register, Christian and Sikh labels rest on a thin classifier training base (Christian ≈2.5%, effectively rural-North-Indian-only for Sikh) and are the highest-risk cells in the name bank.

Pipeline behavior while validation is pending:

- Phase 3 still produces the 50-name-per-cell validation sample (`data/validation/`), ready for annotation whenever it happens.
- `name_bank.csv` carries `human_validated=False` for every Christian and Sikh row until a validation pass is run and κ is computed and recorded.
- Any output, report, or downstream use of these cells must carry an explicit reliability caveat: **labels are classifier/rule-based and unvalidated**. Do not treat Christian/Sikh cell results as equivalent in confidence to Hindu/Muslim results (which have pranaam as an independent cross-check) until validation lands.
- If validation is never run, these cells should be reported with the caveat rather than dropped outright — an unvalidated-but-transparent cell is more useful than silently missing data, as long as the caveat travels with every use of the cell.

## 4. Religion classifiers: unavailable, then partially recovered (2026-09-05 → 2026-09-06)

The plan flagged Phase 3 as "where the actual research risk sits" because of thin classifier training data (§2, risk register). What actually happened was messier, in two stages.

### Stage 1 (2026-09-05): everything looked unusable

- **Chaturvedi's multiclass classifier appeared to have no published trained weights at all.** Its GitHub notebook only ever loaded model files from the author's private Google Drive path (`/content/drive/My Drive/name_to_religion/`). The repo's `models/` directory contains a different task's CNN (USA race prediction) and a 2-class SVM `.sav` with no paired vectorizer file.
- **pranaam (Muslim/non-Muslim binary)** downloaded its model weights from Harvard Dataverse at request time, and that host was 504ing (see `data/PROVENANCE.md`).

**Decision at the time:** don't block Phase 3 — fall back to the method Thorat & Attewell (2007) and Banerjee, Bertrand, Datta & Mullainathan (2009) actually used: explicit, hand-curated marker-name lists (`data/mappings/*_markers.csv`), applied to the real Phase 1/2 frequency data. This is still the primary labelling method — see below for how the classifier now fits in on top of it, not instead of it.

### Stage 2 (2026-09-06): Ojas found the real weights, and we got them working

Once Dataverse recovered, Ojas found that Chaturvedi's classifier weights **were published all along** — just not on GitHub. They're in the paper's official CC0 replication data on Harvard Dataverse (DOI `10.7910/DVN/JOEVPN`, `its_all_in_the_name.zip`, 1.4GB): real `model_multiclass_*.sav` + `vectorizer_multiclass_*.sav` pairs (both SVM and LogisticRegression variants), plus a label encoder.

Getting from "the file exists" to "trustworthy predictions" took another round of debugging:

1. The pickles were made with `scikit-learn==0.22.2.post1` (SVM) / `1.0.2` (LR). Loading either under the project's modern sklearn (1.7.2) doesn't crash — it silently produces garbage. Confirmed by testing: under the version-mismatched load, `khan`, `ansari`, `singh`, `kaur`, `jain` all predicted "Hindu" (the majority class), including names that are completely unambiguous. This is the dangerous failure mode — it looks like it ran successfully.
2. Fixed by building a dedicated environment matching the pickle's actual version exactly: Python 3.8 (via the deadsnakes PPA — Ubuntu 22.04's own repos don't carry it) + `numpy==1.21.6` + `scipy==1.7.3` + `scikit-learn==1.0.2`, isolated in its own venv (`.venv-chaturvedi-lr`), completely separate from the project's main venv.
3. Chose the **LogisticRegression** variant over SVM: `predict_proba()` gives real probabilities (`LinearSVC.decision_function()` is a margin, not a probability, and the plan wants an actual probability threshold), and LR was also more accurate in our own spot-check under correctly-matched versions (`malik`→Muslim and `singh`→Sikh both correct under LR; both wrong under version-matched SVM).
4. Real, confirmed model weaknesses (reproduced under the correctly-matched environment, so these are genuine limitations, not artifacts): `fernandes` predicts Hindu instead of Christian; `ayesha` is a near-coinflip between Muslim and Christian (0.509); most Sikh given names in this project's Punjab data predict Hindu, almost certainly because the classifier's own training data never saw this dataset's unusual transliteration convention (Gurmukhi's implicit vowels preserved rather than dropped — `harpreet` appears as `harapreeth` in this data; see `sikh_firstname_markers.csv`).

**Result:** `src/chaturvedi_classify.py` (run under `.venv-chaturvedi-lr`) scores every `clean_names.csv` candidate and writes `data/processed/chaturvedi_predictions.csv`. `phase3_religion_labeling.py` (run under the normal project venv — it never touches sklearn directly) merges this in as `clf_scores`/`clf_agrees` columns on top of the hand-curated labels. **83% agreement** across the 194 name_bank candidates both methods could score. Disagreements are printed for review rather than auto-resolved either way — several turned out to validate uncertainty we'd already flagged ourselves (the classifier calling `shah` Muslim directly reflects the ambiguity we'd noted for that exact name; calling `kutty` Hindu instead of Christian matches a caveat already written in `christian_surname_markers.csv`).

**Consequence for §3 (hand validation):** unchanged in spirit — `human_validated=False` still applies to every row, because a classifier cross-check is not the same as a human one. But rows now carry a real `clf_scores`/`clf_agrees` signal to prioritize *which* rows most need that human look (start with disagreements).

**pranaam status:** still not usable. Checked its Hugging Face-hosted weights (`gojiberries/pranaam`) too — real files exist, but no released version of the `pranaam` package contains code to load them (see `CITATIONS.md`). Not pursued further since Chaturvedi's classifier already provides a Muslim-vs-rest signal in the meantime.

**Consequence for the Sikh cell:** now included in `name_bank.csv` (27 given names + 2 surnames), now that first names exist. Built as two independently-ranked lists — Punjab-anchored given names and the existing Singh/Kaur surnames — meant to be paired at Phase 5 resume-construction time, per the plan's compound-name design; not a check that any real individual has both, since the frequency tables don't carry joint name pairs.

**Consequence for the Christian cell:** anchored to Kerala and Goa markers only, both first names and surnames. North-East Christian names don't run through a distinctive-name pattern the way Kerala/Goa do (much more tribal/ethnic-name-based) — documented as a gap rather than guessed at with an unreliable marker list.
