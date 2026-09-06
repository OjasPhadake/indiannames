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

### Checked and flagged, initially left alone — then Ojas reviewed and decided (see §8)

A confident wrong "fix" here would have been worse than leaving an uncertain entry alone, so these were reported rather than acted on unilaterally. Ojas reviewed all of them and gave explicit decisions — see §8 for what was actually done with each.

- **Pawar** (Hindu_West) — real data is 94% North, not West. Likely explanation: "Pawar" the Maharashtrian Maratha clan name and "Panwar"/"Pawar" the Rajput clan name common in Rajasthan/UP/MP are different surnames that happen to romanize identically.
- **Shetty** (Hindu_South) — real data is 92% West. Culturally strongly associated with coastal Karnataka; the West-heavy real count may reflect generations of migration to Mumbai, or a spelling collision.
- **Bohra** (Muslim_West) — real data is 95% North.
- **Ansari, Chaudhary** (Muslim_North, Muslim_West) — both real, very large, genuinely multi-region surnames.
- **Singh** (both Hindu_North and Sikh, in Ojas's own list) — not an error, a real and well-documented ambiguity. Left as-is in both places, no decision needed.

## 8. Second QA round: region reassignments for everything flagged in §7 (2026-09-06)

Ojas reviewed every item from §7's automated region-mismatch check (not just the 5 originally reported — a fuller pull turned up 24) and gave an explicit decision for each. All applied to `data/mappings/user_provided_names_raw.py`.

### What moved where

| Name | Was | Now | Basis |
|---|---|---|---|
| Sahni | Christian_North | **Removed from Christian entirely** | Real Punjabi Khatri Hindu/Sikh surname, no Christian association. Already correctly present in this project's own Sikh surname list, independently. |
| Yousuf | Christian_North | **Muslim_South** | Reads more Muslim than Christian in Indian usage; real data 95% South |
| Daniyal | Christian_North | **Muslim_South** | Same reasoning as Yousuf; real data 73% South |
| Wilson | Christian_North | **Christian_South** | Genuinely Christian, wrong region; real data 91% South |
| Isaac | Christian_South | **Christian_North** | Genuinely Christian, wrong region; real data 75% North (opposite direction from Wilson, confirmed independently) |
| Rangwala | Muslim_West | **Muslim_North** | Real data 91% North, literally 0% West |
| Usmani | Muslim_West | **Muslim_North** | Real data 84% North |
| Bohra | Muslim_West | **Muslim_North** | Real data 95% North |
| Jafri | Muslim_North | **Muslim_West** | Real data 89% West |
| Pasha | Muslim_South | **Muslim_West** | Real data 81% West |
| Gazi | Muslim_East | **Muslim_North** | Real data 83% North |
| Sikder | Muslim_Northeast | **Hindu_East** | Religion change, not just region — Sikdar/Sikder is a real Hindu Bengali surname too, per Ojas's cultural knowledge. **Caveat kept on record**: real regional data actually shows this name 78% South, not East/Bengal, likely a spelling collision with an unrelated Southern name — filed as East anyway on Ojas's explicit call, evidence noted rather than silently overridden. |
| Bhatt | Hindu_West only | **Hindu_North + Hindu_West (both)** | Ojas's call — real presence in both (66% North per data, traditional Gujarati-Brahmin association in West) |
| Shetty | Hindu_South only | **Hindu_South + Hindu_West (both)** | Ojas's call — West per real data (92%), South kept for the well-known Karnataka association |
| Ansari | Muslim_North only | **Muslim_East + Muslim_West** (in the source list — see final-output caveat below) | Top-2 by real share: East 73%, West 16% (North is actually 3rd at 10%, so removed from North) |
| Ahmed | Muslim_North only | **Muslim_North + Muslim_East** | Top-2: East 82%, North 11% |
| Chaudhary | Muslim_West only | **Muslim_North + Muslim_East** | Top-2: East 73%, North 27%. Missed adding this to Muslim_North on the first attempt — caught in the verification pass and fixed before this was reported. |

### Two mistakes caught during verification, before reporting

Both found by re-checking the actual pipeline output against what was intended, not just trusting the edit was correct:

1. **Bohra** was removed from `Muslim_West` but the addition to `Muslim_North` was initially forgotten — caught by an absence check, fixed immediately.
2. **Chaudhary** was added to `Muslim_East` but not `Muslim_North` as intended (the "top-2 regions" call needed both) — caught the same way, fixed before this report.

### A structural wrinkle worth understanding: eligibility in the source list is not the same as appearing in the final ranked output

Two things came up that aren't bugs, but need explaining:

- **Ansari ended up in all 4 regions in the final `name_bank_expanded.csv`, not just the intended East+West.** This project's own marker list (`muslim_surname_markers.csv`) also contains "ansari" with no region restriction, so it's an eligible candidate everywhere real frequency supports it — and it does: 264,557 in West and 14,355 in South both clear those regions' top-25 cut on their own merit. This is arguably *more* statistically honest than an artificial 2-region cap would have been, since it's now governed entirely by where the real data actually supports it, not an editorial choice. Flagged here rather than silently allowed to diverge from the literal "top 2" instruction.
- **Several moved/kept names don't appear in the final ranked output at all**, despite being correctly present in the source list: Yousuf, Daniyal, Rangwala, Usmani, Gazi, Sikder, Bohra, and — notably — **Pawar**, which Ojas explicitly said to leave untouched. This isn't a bug or an accidental removal. Selection ranks by a name's real count *specific to that region* (not its national total), and for these names that regional-specific count is small enough to be outranked by ~25-50 other real candidates in the same region. Pawar's national total is large (39,251), but its real West-specific count is only 1,938 — smaller than every other real West Hindu surname in this corpus — so it doesn't crack West's top 25, exactly the same North-collision effect already suspected in §7, just now visible in the ranking rather than just the region-share percentage. Nothing was deleted from the source list; the real data simply doesn't rank these names competitively in their new (or, for Pawar, unchanged) region.

## 9. Third QA round: cross-religion name conflicts found while checking for "optimality" (2026-09-06)

Ojas asked whether the current corpus could be shown to be optimal, or whether further corrections were possible. Three checks were run beyond the two QA passes in §7/§8:

1. **Frequency-floor sensitivity** (`FREQ_FLOOR` 30 → 15 → 5): confirmed the remaining under-target cells (Muslim_F 45/50, Hindu_F 45/50, Christian_F 48/50 pre-fix, Sikh_F 21/50, Sikh_M 43/50, Christian_North surnames 16/25 pre-fix, Sikh surnames 59-64/100) are **hand-list-size-limited, not floor-limited** — lowering the floor to 5 changed only the Sikh surname count (59→64), every other cell was identical at all three floors. There simply aren't more real, distinctively-religious names on either hand list to find at this floor.
2. **Doubled-vowel spelling re-scan** (the same garbage-name pattern that caught rekhaa/meeraa/raajes in §7): came back clean, no new issues.
3. **Cross-religion duplicate check** — grouping the final corpus by `(name, type)` and checking whether more than one religion claims the same first name or surname. This is the one that mattered: a name claimed by two religions defeats the counterfactual swap-study design even when the ambiguity is culturally real, because the same string can't function as a distinctive marker for either group. **Found 6 real conflicts** (beyond the already-documented, deliberately-kept Singh exception from §7).

### Conflicts found and their source

| Name | Type | Conflict | Source of each side |
|---|---|---|---|
| Anita | first, F | Hindu vs Christian | On this project's own Hindu first-name marker list AND on Ojas's Christian_Female list |
| Nisha | first, F | Hindu vs Christian | Same pattern as Anita |
| Patel | surname | Hindu (West) vs Muslim (West) | On this project's own Hindu surname marker list AND on Ojas's Muslim_West list |
| Bhatti | surname | Sikh vs Christian (North) | Pre-existing conflict *within Ojas's own original lists* — on both his Sikh surname list and his Christian_North list |
| Gill | surname | Sikh vs Christian (North) | Same pattern as Bhatti — Gill is on this project's own Sikh marker list too, so it had three-way backing on the Sikh side |
| Kutty | surname | Muslim (South) vs Christian (South) | Pre-existing conflict within Ojas's own original lists — on both his Muslim_South list and his Christian_South list |

Anita/Nisha/Patel were conflicts introduced by merging this project's own marker lists with Ojas's; Bhatti/Gill/Kutty were already present in Ojas's original list before any merging happened.

### Decisions and what was applied

Reported to Ojas with a recommendation for each; he decided:

| Name | Recommended | Ojas's decision | Applied |
|---|---|---|---|
| Anita, Nisha | Remove from Hindu, keep Christian | **Opposite — remove from Christian, keep Hindu** | Removed from `Christian_Female` in `user_provided_names_raw.py` |
| Patel | Remove from Muslim, keep Hindu | **Agreed** | Removed from `Muslim_West` |
| Bhatti, Gill, Kutty | (delegated) | **"Resolve them too, one religion each"** | Bhatti → Sikh, Gill → Sikh, Kutty → Christian_South (my judgment calls, reasoning below) |

**Reasoning for the delegated calls:** Bhatti and Gill were each removed from `Christian_North` and kept as Sikh — both are well-known Punjabi Jat clan names with no real Christian association found in the data, and both were already independently present on this project's own Sikh surname marker list (Gill was on *both* Ojas's own Sikh list and the marker list — three-way backing vs. one weak side). Kutty was removed from `Muslim_South` and kept as Christian_South — it has two-source backing on the Christian side (Ojas's list plus, independently, its well-documented association with Kerala Syrian Christian usage) versus one source on the Muslim side.

### Verification

Re-ran `python src/expand_corpus.py prepare` then `finalize`, then re-ran the same cross-religion duplicate check against the regenerated `name_bank_expanded.csv`: **all 6 conflicts are resolved, zero new conflicts introduced.** The only remaining same-name-different-religion pair in the entire corpus is **Singh** (Hindu_North vs. Sikh) — the pre-existing, deliberately-kept exception documented in §7, left untouched as already decided.

**Direct consequences visible in the final counts** (both are the expected, correct effect of removing names from a cell, not a bug):
- Christian_Female: 48 → 46 (lost Anita, Nisha)
- Christian_North surnames: 16 → 14 (lost Bhatti, Gill)

No other cell changed, since Patel/Bhatti/Gill/Kutty's *other* side (Hindu_West, Sikh, Christian_South) already had a same or better real-frequency candidate filling that slot independently.

**Conclusion on "optimality":** there's no provable global optimum here — no ground truth exists and the objectives (real frequency, religious distinctiveness, hand-list-transparency, corpus size) trade off against each other. But every checkable form of error has now been run to exhaustion: gender-purity mismatches and region mismatches (§7/§8), floor-sensitivity (this round), spelling artifacts (this round), and cross-religion conflicts (this round). Nothing further surfaced. Any future correction would need either new labeled data (for the Sikh classifier gap, per §6) or a new category of check not yet tried.

## 10. Fourth QA round: remaining duplicates + a "does this actually make sense" classifier cross-check (2026-09-06)

Ojas asked three things: (a) are there any remaining repeated names, within his own list or between his and this project's, that §9 missed; (b) do all final names make sense; (c) are they all backed by solid real numbers (he suggested ~10,000 as a rough bar). Three checks were run.

### Check 1: remaining duplicates

Re-ran the same-key and cross-key duplicate scans from §9, this time also checking *within* `FIRST_NAME_BANK`/`REGION_SURNAME_MAP` for a name appearing under two different gender or region keys of the same religion (§9 only checked cross-*religion* duplicates). Found:

- **Mandeep** on both `Sikh_Female` and `Sikh_Male`. Real data: 74% male (n_male=6,213 vs n_female=2,189) — a clear one-gender name, not a genuine unisex case. **Removed from `Sikh_Female`**, kept on Male (already correct there, and the final ranked corpus already only had it as Male — this fixes the source list, doesn't change the deliverable).
- **Gurpreet, Jaspreet, Prabhjot** also appear on both Sikh gender lists — checked and left alone. Real data puts all three inside the project's own 0.4–0.6 ambiguous band (45.7%, 56.1%, 53.8% female respectively), the same genuinely-unisex pattern already documented for Manpreet in §7. Correctly excluded from the ranked output either way; kept on both source lists as an accurate reflection of real ambiguity, not an error.
- Re-ran §9's cross-religion candidate check (merging Ojas's list + this project's marker lists) and found **two new conflicts §9 missed**: **Paul** (Hindu_East vs. Christian_South) and **Biswas** (Hindu_East vs. Muslim_East) — both pre-existing within Ojas's own original list, not introduced by any merge.
- The already-known intentional multi-region duplicates (Bhatt, Shetty, Ahmed, Chaudhary, Ansari — same religion, multiple regions, per §8) are unaffected and correctly left alone. Singh remains the one accepted cross-religion exception (§7).

### Check 2: "does this make sense" — classifier cross-check against a real cultural-plausibility read

Beyond §7's gender/region checks, ran a broader pass: every name where the classifier's independent top pick **disagrees with the assigned religion at ≥85% confidence, excluding "Hindu"** (Hindu is the classifier's known majority-class default bias, documented in §4/#4 — not trustworthy as a signal on its own). 28 names matched. Most were the classifier being wrong in already-understood ways (Bose/Sen called "Christian" — real Bengali Hindu surnames, a Bengali-surname blind spot; Moosa/Hasan/Ravuthar called "Sikh" — the same mohammad→Sikh bias from §4; Mehta called "Jain" — real and true, but Jain isn't one of this study's four target religions, so out of scope; Shah called "Muslim" — a genuine real-world dual-usage case like Singh, but not currently duplicated in the corpus so no action needed). Three, however, held up as real errors:

| Name | Was | Real region data | Classifier | Verdict |
|---|---|---|---|---|
| Momin | Christian_Northeast (98,526 real people, East+West-heavy) | East 57%, West 37% — not Northeast/Garo-population-shaped | Muslim, 95.7% | Genuine two-community spelling collision (see below), not a simple error |
| Yaqub | Christian_North (real n=243, tiny) | East 70%, North only 17% | Muslim, 95.5% | Confirmed error |
| Nazir | Christian_North (real n=38,423) | West 57%, North only 20% | Muslim, 95.1% | Confirmed error |

**Yaqub and Nazir**: both read as Muslim, not Christian, on inspection — Yaqub is the Arabic/Quranic form of Jacob (Indian Christians use "Jacob" itself, already covered by this project's marker lists); Nazir is an Arabic honorific ("observer/overseer") with no standard Biblical-name usage. Neither has an established Christian community association, both were mis-regioned even under their old label (filed North, but real data says East/West), and the classifier agrees at >95% independently. **Applied directly**: removed from `Christian_North`, added to `Muslim_East` (Yaqub) / `Muslim_West` (Nazir) matching their real regional concentration.

**Momin** turned out more interesting on inspection: it sits in `Christian_Northeast` alongside Sangma and Marak — genuine Garo (Meghalaya) Christian tribal clan surnames — so the original placement wasn't arbitrary, it reflects a real Garo Christian surname. But the *national* real count (98,526, concentrated East+West) is far larger than Meghalaya's Garo population could plausibly produce, and doesn't fit a Northeast-only distribution — West India in particular has no Garo population to speak of. The much more numerous explanation is the well-documented Muslim weaver-caste name "Momin"/"Momin Ansari," concentrated in Bengal with westward migration (matches the East+West pattern). **This is a genuine two-community spelling collision, not a data error on either side** — reported to Ojas rather than resolved unilaterally.

### Decisions

| Name | Recommended | Ojas's decision | Applied |
|---|---|---|---|
| Momin | Remove from Christian, keep Muslim | **Agreed** | Removed from `Christian_Northeast` (Sangma/Marak remain as the unambiguous Garo Christian markers), kept on `Muslim_East` |
| Paul | Remove from Hindu, keep Christian | **Agreed** | Removed from `Hindu_East`, kept on `Christian_South` |
| Biswas | Remove from Muslim, keep Hindu | **Agreed** | Removed from `Muslim_East`, kept on `Hindu_East` |

Re-ran `expand_corpus.py prepare`+`finalize` and re-checked: **zero cross-religion conflicts remain except the accepted Singh exception.** Direct consequences: Christian_North surnames dropped 12/25 (from 14, losing Yaqub and Nazir — both were real but too thin/mis-regioned to have mattered much regardless); every other cell unaffected, since the removed side of each conflict already had an equal-or-better real candidate filling its slot.

### Check 3: is everything backed by "solid numbers" (~10,000 real people)?

Checked the full corpus (696 rows) against a 10,000-real-person bar:

| Religion | Rows | n < 10,000 | n < 1,000 | n < 100 |
|---|---|---|---|---|
| Hindu | 195 | 46 | 9 | 0 |
| Muslim | 195 | 88 | 29 | 0 |
| Christian | 183 | 103 | 27 | 1 |
| Sikh | 123 | 95 | 49 | 6 |

**A flat 10,000 floor doesn't make sense for this corpus, and applying one would break the study, not improve it.** India's 2011 census religion shares are roughly Hindu 79.8%, Muslim 14.2%, Christian 2.3%, Sikh 1.7% — a Sikh or Christian name that is genuinely *that community's most common* will still usually have a far smaller absolute national count than an equivalently-ranked Hindu name, purely because the underlying population is ~35-45x smaller. A hard 10,000 cutoff would eliminate the large majority of Sikh entries (77% of Sikh rows) and roughly half of Muslim and Christian entries, while barely touching Hindu — which would make the corpus systematically under-represent minority religions rather than fairly represent them. This project has already been built around exactly this concern (region-specific and religion-specific ranking instead of one global cutoff, `FREQ_FLOOR=30` as a noise floor rather than a popularity floor) — see the module docstring in `src/expand_corpus.py`.

The 7 entries below n=100 were checked individually and are all real: `klair` (30), `yohanan` (43), `khangura` (56), `jawanda` (72), `aujla` (75), `purewal` (81), `sanghera` (85) — six are well-documented real Punjabi Jat/Sikh clan surnames (the same reason Sikh surnames stay pooled nationally rather than split by region, per the module docstring — no single Sikh surname dominates the way Sharma/Singh does for Hindus, so the "top 100" pool necessarily reaches into thinner names), and Yohanan is a real, if rare, Hebrew-derived Indian Christian name (the Hebrew form of John). All seven clear the project's own `FREQ_FLOOR=30` noise floor and have no gender or region mismatch. No further action taken — this is what an honest, population-proportional corpus looks like for minority religions in India, not a data quality problem.

## 11. A real pipeline bug found while raising the Sikh surname floor, plus a fifth QA round (2026-09-06)

Ojas pushed back on the six sub-100 Sikh surnames from §9/§10 (`klair`, `khangura`, `jawanda`, `aujla`, `purewal`, `sanghera`) — asked whether they could be swapped for more common real alternatives, and whether this project already had something documenting popular Sikh surnames to check against. It didn't (§6/§9 already established the classifier can't be trusted for Sikh surnames), so the check was done by hand: pull real national frequencies for well-known Jat/Khatri Sikh clan surnames not yet on either hand list, and see which ones are both real and not already claimed elsewhere.

### The pipeline bug

While checking `mann`, `walia`, `uppal`, `khaira`, `gosal`, `heer`, `mahil`, and `natt` — all already on Ojas's own Sikh list, all with real backing from 88 (`natt`) up to 11,966 (`mann`) — none of them showed up anywhere in `name_bank_expanded.csv`. Traced it to `build_canonical_map()` in `src/expand_corpus.py`: the trailing-"a"-strip merge rule (§the module's own transliteration_stem, previously known to be risky for gendered first-name pairs) was silently renaming these surnames to a *different, unrelated* real word that happens to share the same stem and has a bigger national count — `mann`→`manna`, `walia`→`wali`, `uppal`→`uppala`, `khaira`→`khair`, `gosal`→`gosala`, `heer`→`heera`, `mahil`→`mahila`, `natt`→`natta`. The candidate-matching step downstream still looked for the *original* spelling, which no longer existed anywhere in the frequency table after the rename, so the name vanished from the corpus with no error, warning, or row of any kind.

**This wasn't Sikh-specific.** A full scan of every candidate surname across all four religions against this same failure mode found **18 real, hand-listed candidates affected**, 16 of which had silently vanished entirely: `banerjee` (Hindu, 17,782 real people — one of the most recognizable Bengali Hindu surnames in the corpus), `masih` (Christian, 5,413 — the *first* name on the `Christian_North` list), `baig` (Muslim, 9,533), `vyas` (Hindu, 19,994), `bhargava` (Hindu, 2,708), `sultan`, `chang`, `chana`, `kapadia`, `chetia`, plus the 8 Sikh ones above. Two (`sultan`, `chana`) happened to survive by chance. This had been silently understating four religions' real corpuses since the very first `expand_corpus.py` run in §5 — not something introduced this round.

**Fix applied** (`src/expand_corpus.py`, `build_canonical_map()`): added a `protect` parameter — any name that's on a hand list (any religion, ours or Ojas's) now always keeps its own spelling and its own real count through the merge, instead of being eligible to be silently absorbed into a bigger, unrelated stem-mate. An *unprotected* spelling (real transliteration noise nobody explicitly curated, e.g. doubled-vowel artifacts) still merges into the group's biggest member exactly as before — this only stops a **curated** candidate from disappearing, it doesn't weaken the original de-duplication for genuine noise. Threaded through `load_freq_tables()` (surnames, protected set = every hand-list surname across all four religions) and `select_firstnames()` (first names, protected set = that call's own hand-list candidates).

**Verified impact after the fix** (re-ran `prepare` + `finalize`): all 8 previously-vanished Sikh surnames recovered with their real counts (`mann` 11,966 down to `natt` 244). `banerjee`, `baig`, `vyas`, `masih`, `kapadia` also recovered under their own spelling; `bhargava`, `chang`, `chetia` recovered too but their real region-specific count doesn't clear their cell's top-25 (same legitimate "eligibility ≠ inclusion" pattern as §8 — not a new problem). Net effect across the whole corpus: Hindu_Female first names hit 50/50 for the first time (was 45/50), Muslim_Female 45→47, Christian_Female 46→47, Christian_North surnames 12→13, and **Sikh surnames 59→67/100**.

### Raising the Sikh surname floor

With the bug fixed and 8 strong real replacements already recovered, removed the 6 named weak entries per Ojas's ask — all real (30-85 people nationally) but thinner than what's now available:

| Removed | n | Replaced in spirit by (already recovered above) | n |
|---|---|---|---|
| Klair | 30 | Mann | 11,966 |
| Khangura | 56 | Walia | 9,176 |
| Jawanda | 72 | Uppal | 5,178 |
| Aujla | 75 | Khaira | 4,203 |
| Purewal | 81 | Heer | 1,316 |
| Sanghera | 85 | Gosal | 1,064 |

Not a literal one-to-one substitution — these 8 were already recovered by the bug fix and would have stayed in the corpus regardless; removing the 6 weakest just tightens what's left. Final pooled Sikh surname count: **61/100, floor raised from n=30 to n=106** (`nagra`). Re-ran the full cross-religion conflict check afterward: still just the one accepted Singh exception, nothing new introduced.

**Why not fill all the way to 100 with more candidates instead of just removing the weak 6:** the same conclusion as §6/§9 still holds — the hand lists (Ojas's + this project's own) are now close to exhausted for Sikh surnames specifically, and the classifier can't safely fill the rest (§6). Going from 61 to 100 would need either new labeled training data or a slower, fully-manual research pass through Punjabi genealogical/clan-name sources beyond what's already been checked here — flagged as a possible future step, not attempted in this round.
