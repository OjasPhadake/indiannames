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

## 4. Religion classifiers turned out to be unavailable, not just risky (2026-09-05)

The plan flagged Phase 3 as "where the actual research risk sits" because of thin classifier training data (§2, risk register). What actually happened during Phase 3 was more basic: **every classifier the plan named was unusable when we went to call it**, for two different reasons:

- **Chaturvedi's multiclass classifier has no published trained weights at all.** Its notebook only ever loaded model files from the author's private Google Drive path (`/content/drive/My Drive/name_to_religion/`). The GitHub repo's `models/` directory contains a different task's CNN (USA race prediction) and a 2-class SVM `.sav` with no paired vectorizer file. This is permanent, not an outage — the asset was never released publicly.
- **pranaam (Muslim/non-Muslim binary) is a real, current package**, but as of 2026-09-05 it downloads its model weights from Harvard Dataverse at request time, and that host is the same one that's been 504ing since Phase 1 (see `data/PROVENANCE.md`). Temporarily unavailable, tracked by the same recovery routine as naampy's first-name data.

**Decision:** rather than block Phase 3 entirely, `src/phase3_religion_labeling.py` falls back to the same method Thorat & Attewell (2007) and Banerjee, Bertrand, Datta & Mullainathan (2009) actually used — explicit, hand-curated marker-surname lists (`data/mappings/muslim_surname_markers.csv`, `hindu_upper_caste_surnames.csv`, `christian_surname_markers.csv`, `sikh_surname_candidates.csv`), applied to the real Phase 1/2 frequency data. Counts are real electoral-roll counts; category assignment is manual, not model-inferred.

**Consequence for §3 (hand validation):** the human_validated=False caveat from decision #3 above no longer applies only to Christian/Sikh — it now applies to every row in `name_bank.csv`, Muslim and Hindu included, because none of it has been through a classifier cross-check yet, only a hand-curated list. Re-run Phase 3 once pranaam's weights are reachable and treat its output as a check against the current curated lists, not a replacement that can be skipped.

**Consequence for the Sikh cell:** not included in `name_bank.csv` yet at all (as opposed to included-with-caveat like Christian). The plan is explicit that Singh/Kaur alone can't separate Sikh from Hindu Rajput/the wider Hindi belt without pairing with a validated Punjabi Sikh given name — and first names are still blocked on the same Dataverse outage. Surname-only candidates are staged in `data/processed/sikh_surname_candidates_staged.csv` for when that becomes possible.

**Consequence for the Christian cell:** anchored to Kerala and Goa markers only. North-East Christian surnames don't run through a distinctive-surname pattern the way Kerala/Goa do (much more tribal/ethnic-name-based) — documented as a gap rather than guessed at with an unreliable marker list.
