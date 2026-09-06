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

## 5. Expanded corpus (~100 names/religion) for Ojas's own tool — a second output, not a replacement (2026-09-06)

Ojas's own downstream tool needs bigger cells than this project's ~25/cell design — roughly 100 first names and 100 surnames per religion — and already had a hand-typed reference (`FIRST_NAME_BANK`/`REGION_SURNAME_MAP`, ~40-90 names/cell) that he wanted rebuilt from real statistics instead. Explicit request: merge three sources — his list, this project's marker lists, and the Chaturvedi classifier — "best of all 3."

**What actually happened when all three were merged:** the classifier as a *candidate source* (rather than a cross-check) was tried at two thresholds and rejected at both.

- At `clf_max_prob >= 0.5`: real, confirmed errors got in with zero cross-check to catch them. `mohammad` and `hussain` — the exact `mohammad`→Sikh misclassification already documented in §4 above — showed up as **Sikh surnames**, imported into the corpus instead of just noted as a known weakness.
- At `clf_max_prob >= 0.85`: the same failure mode resurfaced on different spellings. `moohammad` (a raw spelling variant, 90.7% confidence) and `sulaiman`/`moosa` (unambiguously Muslim names) still landed as Sikh surnames; `varun`/`sen` (Hindu-pattern names) landed as Christian surnames at 90%+ confidence. This isn't a threshold-tuning problem — the classifier has a specific, repeatable bias toward mislabeling unfamiliar-looking tokens as Sikh (and to a lesser extent Christian) that no single probability cutoff reliably screens out.

**Decision: classifier-only candidates are not used at all.** Tested disabling the classifier as a candidate source entirely (keeping only hand-list-backed candidates, ranked and floored by real frequency): Muslim/Hindu/Christian still reached 94-100/religion without it, so the safety trade cost almost nothing. The classifier is still run and merged into the output as `clf_predicted_religion`/`clf_max_prob` on every row — same cross-check role it plays in `name_bank.csv` — just not as a source of new names. Cross-check agreement on this corpus is lower than `name_bank.csv`'s 83% (53% here), which is expected, not alarming: this corpus deliberately reaches into more regional/community-specific names than the classifier's own (thin, documented) training data represents well.

**Other gates actually lowered, and why each is safe:**
- **Frequency floor 30** (was Phase 2's 500) — real names Phase 2's stricter floor would have cut, not noise. Still excludes single/double-digit counts.
- **Gender purity 0.6/0.4** (was 0.95/0.05) — names in the 0.4-0.6 band are still excluded outright as genuinely ambiguous, not forced into a bucket.
- **Region split extended to Muslim and Christian surnames** (previously only Hindu had a region breakdown) — same 4 regions as everywhere else in this project (Northeast folded into East), per Ojas's own explicit preference over adding a 5th bucket.
- **Transliteration-stem merge applied to Phase 1's raw output** (`freq_firstnames.csv`/`freq_surnames.csv`, which unlike `clean_names.csv` haven't been through Phase 2's own dedup) — without it, doubled-vowel spelling variants of ordinary names (`rekhaa`, `meeraa`, `sulekhaa` — real spellings of Rekha/Meera/Sulekha in this dataset) surfaced as spurious "new" candidates. Done *within* gender bucket for first names specifically — merging before splitting by gender would have wrongly conflated real male/female name pairs like arun/aruna.

**A real, surprising, statistically-driven finding this approach is supposed to produce:** "Sana" comes out 72% male nationally in the electoral-roll data (`prop_female=0.28`), landing it in `Muslim_Male` rather than the `Muslim_Female` bucket Ojas's own hand list had it in. Left as the real data says, not corrected toward the hand-list assumption — that's the entire point of grounding this in statistics rather than typing names from memory. Worth double-checking if it matters for the downstream use, since it could also be a transliteration collision with an unrelated male name rather than the same name skewing male.

**Achieved counts** (`data/processed/name_bank_expanded.csv`, 693 rows / 588 unique names): Hindu 95 first + 100 surnames, Muslim 94 + 100, Christian 97 + 94, Sikh 54 + 59. Sikh and Christian-North fall short of 100 honestly rather than being padded — thinner real candidate pools, not a bug. Output also provided pre-formatted to match Ojas's original dict structure (`data/processed/name_bank_expanded_dict_format.py`) for a direct drop-in.

**Kept as a second output, not merged into `name_bank.csv`:** the two serve different purposes — `name_bank.csv` is this project's own tightly-validated ~25/cell primary output (plan §3.3), `name_bank_expanded.csv` is Ojas's higher-volume, lower-floor corpus for his own tool. Conflating them would blur which validation standard applies to which row.

## 6. Two attempts to make religion-labelling less hand-list-dependent — both tested, both rejected (2026-09-06)

Ojas asked directly: can this rely more on the electoral-roll statistics and less on hand-typed lists? Two standard techniques were built and actually tested against the known error cases (`moohammad`, `sulaiman`, `moosa`, `hussain`, `abdullah`, `ahammad`, `naushad` → wrongly "Sikh"; `varun`, `sen` → wrongly "Christian" as surnames) rather than assumed to help from the formula alone. Both failed, for a related structural reason. `src/prior_correct.py` and `src/soft_score.py` are kept in the repo as working, documented tools — this is a negative result worth having on record, not code to delete.

**Attempt 1 — prior correction** (`src/prior_correct.py`). The classifier's training data is 84.5% Hindu, 9.1% Muslim, 3.2% Sikh, 2.45% Christian (real counts from `REDS_train_multiclass.csv`, saved to `data/mappings/chaturvedi_train_class_prior.csv` since the raw REDS file isn't in this repo) — so it defaults to guessing Hindu when unsure. Standard fix: divide each class's probability by its training share, multiply by a fairer target share, renormalize. Tested two targets on the full ~5,650-name broad pool used earlier:

- **Uniform target** (25% each across our 4 study religions): changed **43% of all 5,652 predictions** — far too blunt an instrument to trust. The Sikh bucket grew from 27 to 125 names, and got *worse*, not better: `abdullah`, `fathima`, `hussain`, `ibrahim` and more clearly-Muslim names flooded in, because boosting Sikh's prior 8x amplifies exactly the wrong high-confidence guesses along with any right ones. It also broke a previously-correct case (`iyer`, a real Hindu surname, flipped to Christian).
- **Census target** (India's real ~1.7% Sikh, ~14.2% Muslim population share): far more conservative, only 149/5,652 changed — but of the 8 known Sikh-error names, only 1 (`naushad`) actually moved away from Sikh (to Hindu, still wrong). The other 7 stayed confidently Sikh; their probability dropped somewhat but never enough to flip the label.

**Verdict: rejected.** Neither target reliably fixes the errors we already know about, and the aggressive one actively multiplies them. This makes sense in hindsight: prior correction only fixes "defaults to the biggest category when unsure" — a calibration problem. Our actual problem is different: the classifier is *specifically and confidently* wrong about certain names, most likely because it never saw this dataset's transliteration convention (Gurmukhi's implicit vowels preserved rather than dropped) during training. That's a training-data-coverage problem, and no amount of output rescaling touches it.

**Attempt 2 — soft-scoring** (`src/soft_score.py`). Instead of a hard "on a hand list or excluded" gate, score every name as `0.7 × hand_list_membership + 0.3 × raw_classifier_probability` and rank by that combined score, letting a high-confidence classifier-only name compete for a spot rather than being excluded outright. Ran it end to end and checked the actual output, not just the formula:

- Muslim, Hindu, and Christian barely changed — hand lists alone already fill nearly all 50/region-or-gender slots, so there's almost no room for a classifier-only name to compete for anyway.
- **Sikh surnames is where it mattered, and it went badly.** Real hand-list-backed Sikh surnames only fill ~65-72 of the 100 target slots. The remaining ~28-35 got filled by the classifier's *next-most-confident* guesses — which turned out to be exactly the same wrong names as before (`moosa` 0.94, `moohammad` 0.91, `sulaiman` 0.86, `abdullah` 0.79, `hussain` 0.79, `ahammad` 0.77, `naushad` 0.54), because the classifier doesn't have any *better* Sikh guesses left once the real ones are accounted for.

**Verdict: rejected**, for a structural reason worth stating plainly: **soft-scoring only helps a thin cell if the classifier has good uncounted candidates left for that cell — and for Sikh specifically, it doesn't.** Lowering the hand-list weight further would only invite more of the same names in with higher scores, not better ones. This isn't a tuning problem; it's a ceiling on what this particular classifier knows about Sikh names.

**What this means going forward:** the current design — hand lists gate membership, the classifier only cross-checks and never adds names — isn't a cautious placeholder waiting to be replaced by something more statistical. It's the actual right call given what's been tested. Getting genuinely more data-driven for the Sikh cell specifically would need a classifier that's actually seen this dataset's transliteration convention during its own training — i.e. new labeled data, not a smarter way to use the existing model's output.

## 7. One-by-one QA pass on both hand lists against real electoral-roll data (2026-09-06)

Ojas asked for every name in both hand lists (his `user_provided_names_raw.py` and this project's own marker lists) to be checked individually, with confirmed errors replaced by a real, statistically-verified, correctly-labelled alternative. Two objective checks were run against every first name and surname, plus a manual review — not just skimmed, actually checked against real data and re-verified with the classifier as a second opinion before touching anything.

**Method:** for every first name, compared the list's claimed gender against the real male/female split in the electoral-roll data. For every surname, compared the claimed region against where the surname is actually most concentrated in real data. Anything flagged was cross-checked against the (raw, uncorrected) classifier's independent opinion before being treated as confirmed, and a replacement candidate was only accepted after its own real frequency, real gender purity, and classifier opinion all lined up — the same discipline as every other fix in this project, not a one-off exception.

### Confirmed and fixed (6 changes, all in `data/mappings/user_provided_names_raw.py`, each marked inline)

| Was | Religion/gender/region claimed | Real data says | Replaced with | Real data for replacement |
|---|---|---|---|---|
| Sana | Muslim_Female | 72% male (n=17,009) | **Shahida** | 99% female, n=14,300, classifier: Muslim 88% |
| Rosario | Christian_Female | 93% male (n=1,069) | **Flory** | 99% female, n=549, classifier: Christian |
| Anoop | Christian_Male | 96% North-concentrated, no Christian association; classifier independently calls it Hindu (47%) | **Justin** | 99% male, n=5,023, classifier: Christian |
| Prasad | Hindu_South (surname) | 70% North-concentrated — a generic pan-Indian name, not distinctively Southern | **Krishnan** | n=591,010 in South, classifier: Hindu 97%, genuinely Tamil/South-distinctive |
| 9 names* | Sikh_Female | 66-96% male in real data (not the near-50/50 genuinely-unisex pattern most of this list actually is) | moved to Sikh_Male | — |
| Gurinder | Sikh_Male | 66% female | moved to Sikh_Female | — |
| Manpreet | Sikh_Male (duplicate) | 70% female | removed from Male (already correctly on Female) | — |

*Kuldeep, Manveer, Rajdeep, Ravneet, Rajinder, Ravinder, Jasveer, Parminder, Navneet.

Interesting confirmation from the fix itself: "Sana" is *still* in the final corpus after this change — now correctly placed as **Muslim Male** (this project's own marker list has it too, with no gender pre-assigned, so the real data placed it correctly on its own). Nothing was lost; it just moved to where it actually belongs. The Sikh gender corrections were a net win beyond just accuracy — Sikh first names in `name_bank_expanded.csv` went from 54 to 64 as a direct result, since several of the moved names cleared the real-frequency floor as Male when they hadn't been eligible as Female.

This project's own marker lists (`data/mappings/*_markers.csv`) were checked the same way and came back clean — no confirmed errors found in first-name gender (they were never gender-tagged to begin with, so there was nothing to contradict) or surname/religion pairing.

### Checked and flagged, but NOT changed — evidence was ambiguous, not confirmed-wrong

A confident wrong "fix" here would be worse than leaving an uncertain entry alone, so these are reported rather than acted on:

- **Pawar** (Hindu_West) — real data is 94% North, not West. Likely explanation: "Pawar" the Maharashtrian Maratha clan name and "Panwar"/"Pawar" the Rajput clan name common in Rajasthan/UP/MP are different surnames that happen to romanize identically — the aggregate count probably blends two real, distinct surnames rather than mislabelling one.
- **Shetty** (Hindu_South) — real data is 92% West. Shetty is culturally strongly associated with coastal Karnataka; the West-heavy real count may reflect generations of migration to Mumbai, a genuine regional data quirk rather than a labelling error, or (less likely) a spelling collision. Not confident enough to touch.
- **Bohra** (Muslim_West) — real data is 95% North. The Dawoodi Bohra community is historically Gujarat/Mumbai-based, but "Bohra" as a bare surname string may also be used by unrelated communities elsewhere, especially at this scale (aggregate counts don't carry community context).
- **Ansari, Chaudhary** (Muslim_North, Muslim_West) — both are real, very large, genuinely multi-region surnames (Ansari: 1.65M total, Chaudhary: 1.2M total) that happen to have their single biggest concentration in East rather than the claimed region. Not wrong so much as under-specified by a single-region tag — these names are legitimately common in several regions at once.
- **Singh** (both Hindu_North and Sikh, in Ojas's own list) — not an error, a real and well-documented ambiguity (used by Hindu Rajputs and other communities, not just Sikhs). Already flagged in conversation when first noticed; left as-is in both places since the list creator put it there deliberately in both, correctly reflecting real-world overlap.
