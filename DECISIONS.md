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
