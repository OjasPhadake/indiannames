# Counterfactual Resume Audit of LLM Hiring Bias — India

**Building a religion × gender × region name bank and the evaluation harness around it**

---

## 1. What this document covers

A build plan for two artifacts:

1. **`name_bank.csv`** — first names and surnames for India, frequency-ranked, labelled by religion (Hindu / Muslim / Christian / Sikh), gender, and region, with provenance and confidence for every row.
2. **A counterfactual ranking harness** — synthetic resumes held constant while the name is swapped, measuring average rank change across LLM rankers.

The name bank does not exist as a published dataset. Region and gender are available from electoral-roll data at real frequency weights; religion is only available through classifiers. Phase 3 is where the actual research risk sits.

---

## 2. Source inventory

### Frequency data (region + gender)

| Source | What it gives | Notes |
|---|---|---|
| `appeler/naampy` | First name × state × birth-year, with `n_male`, `n_female`, `n_third_gender`, `prop_female` | Built on parsed Indian electoral rolls. Ships raw counts, so it can be sorted by frequency per state. ~30 states/UTs. |
| `appeler/instate` | Surname × state distribution, with `total_n` | 2017 electoral rolls, 34 states/UTs in the v2 table. Ships `lastname_langs_india.csv` inside the package. |
| Sood & Dhingra corpus (arXiv 2303.06823) | ~438M voter records, 33 states, ~1.14M unique surname spellings | The corpus underlying both packages. Full data: Harvard Dataverse `10.7910/DVN/ZXMVTJ`. |

### Religion labelling (classifiers only — no frequency tables exist)

| Source | Classes | Training data | Limitation |
|---|---|---|---|
| `RochanaChaturvedi/it-is-all-in-the-name` (arXiv 2010.14479) | Hindu, Muslim, Christian, Sikh, Jain, Buddhist (`n_way="multiclass"`) | NCAER REDS + 20k hand-annotated rural UP household heads | Christian ≈ 2.5% and Buddhist ≈ 0.3% of training observations. Christian and Sikh classes rest on a thin, rural, North-Indian base. |
| `appeler/pranaam` | Muslim / not-Muslim only | Bihar Land Records, ~4M unique records | Binary. ~98% OOS accuracy. Bihar-only training data. Use as a cross-check, not a primary labeller. |
| `raphael-susewind/name2community` (Field Methods 2015) | Religious community | Dictionary matching | Requires generating your own master name list first. Low coverage; cannot classify unseen names or spelling variants. |

### Method precedent (cite, don't mine for names)

- **Thorat & Attewell (2007)** — 4,808 applications, names identifiable as upper-caste Hindu, Dalit, or Muslim. Hand-picked stereotypes, a handful per cell.
- **Banerjee, Bertrand, Datta & Mullainathan (2009)** — 3,160 resumes; caste-linked surnames (Sharma, Bhatia, Aggarwal / Paswan, Manjhi, Pasi).
- **"Invisible Filters" (AIES 2025)** — closest prior LLM work. Varies names by gender, caste, region within Indian interview transcripts; finds **no significant** effect of region or caste, marginal gender effects that fail Tukey post-hoc. 50 transcripts × 8 conditions = 400 points. Likely underpowered — read before finalising the design.

### Known dead end

- **Vahini et al. (arXiv 2209.03089)** — gender + caste over 7.63M unique names. No religion axis, and they release the *scraping code*, not the datasets, as a privacy policy. Do not budget time for this.

---

## 3. Design decisions (settle before writing code)

### 3.1 Factor structure

- **Primary:** religion (4) × gender (2) = 8 cells, fully crossed.
- **Secondary:** region (N/E/W/S) nested **inside Hindu only**, reported as a separate sub-study.
- **Optional third:** city on the resume, manipulated *independently* of the name, so "name signals region" can be separated from "address signals region".

**Why not a full 4×4×2 grid:** religion and region are heavily confounded in India. South Indian Sikh names, West Indian Christian names outside Goa, and similar cells barely exist in the population. Any "most common name" assigned to them is a classifier artifact, and a crossed design conflates religion with region.

### 3.2 Caste inside the Hindu cell

Sharma and Paswan are both Hindu and behave very differently in the audit literature. **Decide explicitly:** hold caste constant (upper-caste surnames only, matching Thorat & Attewell's baseline) or open it as a fourth axis. Leaving it uncontrolled makes "Hindu" an average over an uncontrolled variable.

### 3.3 Cell size

Target **25 first names + 25 surnames per cell**. Below ~15 the finding is a property of five specific names, not of the category.

### 3.4 Metrics and model, fixed in advance

- Mean rank change Δ per swap (primary), normalized rank, pairwise win rate.
- Linear mixed-effects: religion + gender + region as fixed effects; resume ID and slate ID as random effects; name-familiarity covariates included.
- **Run a power calculation against the Invisible Filters null before collecting anything.** Know what effect size your N can detect.

---

## 4. Build phases

### Phase 1 — Frequency base (region × gender)

1. `pip install naampy instate`. Pull full tables from Harvard Dataverse if the packaged subsets are too small.
2. Extract first name × state × (`n_male`, `n_female`, `prop_female`) from naampy; surname × state with `total_n` from instate.
3. Write an explicit **state → region mapping file**. Decide where the North-East, Odisha, and the central/Hindi-belt states go. Do not bury this in a dict inside a script.
4. Aggregate to region level. Raw voter counts already reflect population, so raw is usually correct — document the choice either way.

**Output:** `freq_firstnames.csv`, `freq_surnames.csv`

### Phase 2 — Clean and filter

Electoral-roll name fields are noisy; this phase matters more than it sounds.

- **Normalize:** lowercase; strip honorifics and relational tokens (`smt`, `shri`, `d/o`, `w/o`, trailing `devi` used as a marker); collapse transliteration variants (`patila`→`patil`, `dasa`→`das`, `-ee`/`-i` endings).
- **Frequency floor:** minimum n per region — start at 500, tune.
- **Gender purity:** keep first names with `prop_female` ≥ 0.95 or ≤ 0.05. Ambiguous names silently weaken the gender manipulation.
- **Deduplicate near-identical spellings** so "top 25" isn't five spellings of one name.

**Output:** `clean_names.csv`

### Phase 3 — Religion labelling

1. Run surviving names through Chaturvedi multiclass **and** pranaam binary.
2. Keep rows where both agree on the Muslim / non-Muslim split **and** Chaturvedi's max-class probability clears a threshold (start at 0.9).
3. **Sikh cell — rule-based, not classifier-based.** Require Kaur (F) or Singh (M) as surname *plus* a validated Punjabi Sikh given name (Harpreet, Gurmeet, Jaspreet, Manjit…). Build that given-name list from naampy's Punjab table, top-ranked, then hand-check. Singh alone does not separate Sikh from Hindu Rajput or the wider Hindi belt.
4. **Christian cell — region-anchored.** Pull top-frequency surnames from instate restricted to Kerala (Thomas, Varghese, Mathew), Goa (Fernandes, D'Souza), and the North-East, then intersect with Chaturvedi's Christian predictions. Treat the classifier as a filter, not a source.
5. **Hand validation.** Sample 50 names per cell, two annotators, report Cohen's κ. Any cell below κ ≈ 0.8 gets fixed or dropped. Reviewers will ask about this step specifically.

**Output:** `name_bank.csv` — `name, type (first/last), gender, religion, region, n, source, clf_scores, human_validated`

### Phase 4 — Name-familiarity control

LLMs penalize rare and orthographically unusual tokens independently of demographics. If the Muslim cell has longer, rarer names than the Hindu cell, the experiment measures tokenization rather than bias.

Match cells on:

- character length
- token count under the target tokenizer
- number of name tokens
- log frequency (`n`)

Do this by stratified sampling **and** include the same variables as covariates in the mixed model. Both, if possible.

### Phase 5 — Resume generation

- Generate identity-neutral base resumes across 3–4 job families × 2–3 quality tiers. The base set stays fixed across all conditions.
- **Leakage audit before swapping:** email handles, university names implying region, mother-tongue fields, addresses, referee names.
- Substitute only the name (and city, if manipulated). Store each resume with its full condition tuple so counterfactual pairing is explicit in the data, not reconstructed later.

### Phase 6 — Ranking harness

- **Slate design:** single-swap slates (one candidate's name perturbed, the other 19 or 49 fixed) as primary. All-swapped slates as a secondary check.
- **Randomize position** within the slate; run multiple orderings per slate. Position effects in LLM ranking are large and will otherwise swamp the signal.
- Temperature 0 where available, plus a sampled-temperature replication. Multiple models, multiple prompt phrasings.
- Log: model version, prompt hash, seed, slate composition, raw output, parsed rank.

### Phase 7 — Analysis

- Mixed-effects on Δrank as specified in §3.4.
- Multiple-comparison correction across cells.
- Report effect sizes and confidence intervals, not just p-values.
- **Report null cells as prominently as significant ones.** Given the Invisible Filters result, a well-powered null is a publishable finding.

---

## 5. Open decisions blocking Phase 1

1. **State → region mapping** — specifically the North-East and the central states.
2. **Caste inside the Hindu cell** — held constant, or opened as a fourth axis?
3. **Hand validation** — are you willing to annotate the Christian and Sikh cells? If not, those cells should be reported with an explicit reliability caveat or dropped.

See [`DECISIONS.md`](../DECISIONS.md) for how these were resolved.

---

## 6. Risk register

| Risk | Mitigation |
|---|---|
| Christian / Sikh labels unreliable (thin classifier training base) | Rule-based + region-anchored construction, plus mandatory hand validation with reported κ |
| Religion–region confounding | Nested design, not crossed; region studied inside Hindu only |
| Name-familiarity confound masquerading as bias | Phase 4 matching + covariates |
| Position effects in slate ranking | Randomized ordering, multiple orderings per slate |
| Underpowered design returning an uninformative null | Power calculation before data collection; ≥25 names per cell |
| Singh ambiguity collapsing the Sikh/Hindu contrast | First-name + surname compounds, never surname alone |

---

## 7. Ethics and release

The upstream data is derived from Indian electoral rolls, released for research use with privacy conditions attached. Release the **name bank and code**, not any linkage to individual voter records. Vahini et al.'s precedent — publishing collection code rather than raw name datasets — is the conservative norm in this literature and worth following. State clearly in any paper that classifier-inferred religion labels are aggregate-pattern estimates and must not be used for individual classification; both pranaam and naampy carry that warning in their own documentation.

---

## 8. Suggested order of work

**Phases 1–3 are mechanical** and can be built immediately once §5 is settled; the deliverable is `name_bank.csv` plus the hand-validation sample pulled out and ready for annotation. Phase 4 depends on the target tokenizer. Phases 5–7 depend on nothing in 1–4 and can be prototyped in parallel using placeholder names.
