# envs_tab: knowledge-required real-data environments (InfoSeek)

Fix for reviewer R1 **M1** ("calculator"): the model sees **only** check descriptions, costs, payoffs and the empirical
TRAIN base rate. No likelihood or reliability is stated anywhere in `text`. The gold decision values come from a reference
model fit on real training records (no LLM is involved). M5 is covered as well: DP gold, ties, and non-myopic flags are stored,
and the natural (unstratified) distribution is the primary file. M7 is covered by out-of-sample AUC, calibration and
realized-utility checks (below).

## Datasets (UCI; raw files in `raw/<ds>/`, read with `python -I`)
| ds | source | domain | Y (adverse = 1) | n | train / cal / test | train base rate |
|---|---|---|---|---|---|---|
| taiwan | UCI 350, default of credit card clients | credit | default next month | 30,000 | 18,000 / 6,000 / 6,000 | 0.221 |
| cdc | UCI 891, CDC Diabetes Health Indicators (BRFSS 2015) | clinical | diabetes or prediabetes | 253,680 | 152,208 / 50,736 / 50,736 | 0.139 |
| german | UCI 144, Statlog German credit | credit | bad credit risk | 1,000 | 600 / 200 / 200 | 0.300 |
| heart | UCI 45, Heart Disease (Cleveland, Hungary, Switzerland, VA) | clinical | angiographic disease (num > 0) | 920 | 552 / 184 / 184 | 0.553 |

The splits are stratified 60/20/20 with fixed seeds (`prep.py:SPLIT_SEED`). **dev** scenarios are built from CAL records
and **test** scenarios from TEST records.

Actions (envs_synth style, act = risky action):
- taiwan: APPROVE / DECLINE (credit-line review).
- german: APPROVE / DECLINE (consumer loan).
- cdc: SKIP / REFER (diabetes testing referral).
- heart: DISCHARGE / REFER (coronary angiography).

Payoffs: U(act, Y=0) = +g, U(act, Y=1) = -l, U(pass, ·) = 0, with g in {1,2,3} and l in {2,3,5,8}. Costs are
regime x min(g,l) x mult, with regime in {0, .05, .1, .2, .4} and mult in {.5, 1, 1, 1.5, 2} (as in envs_synth).

## Findings (binarization fixed a priori from domain conventions and column docs; see `scripts/features.py`)
Placebos are chosen from a pre-declared **candidate pool** by the empirical rule |corr(f,Y)| < 0.02 on TRAIN. The rule
kept up to 2 per dataset, those with the smallest |corr|. Heart had no candidate passing the strict rule. Its placebo
therefore uses a documented small-n fallback: a record-keeping attribute from the 76-attribute files with p_train > 0.3
and |corr_all| < 0.02. This attribute is missing for Cleveland records, so heart has a placebo in only ~1/3 of scenarios.
emp s1 / emp s0 are the naive-Bayes P(f=1|Y=1) and P(f=1|Y=0) on TRAIN. They are stored per check for diagnostics and
are **never shown** to the model.

**taiwan** (n=30000, train base rate 0.221)

| key | check name | finding text | threshold | placebo | prev | corr(f,Y) train | emp s1 | emp s0 | missing |
|---|---|---|---|---|---|---|---|---|---|
| late2_latest | Latest repayment status | the client's most recent bill is two or more months overdue | `X6 (PAY_0) >= 2` |  | 0.106 | +0.393 | 0.332 | 0.041 | 0.00 |
| late_prev | Previous month's repayment status | the client was at least one month late on the previous month's bill | `X7 (PAY_2) >= 1` |  | 0.149 | +0.341 | 0.377 | 0.085 | 0.00 |
| late3_of6 | Six-month delinquency count | the client was late on three or more of the last six monthly bills | `#{X6..X11 >= 1} >= 3` |  | 0.127 | +0.350 | 0.345 | 0.064 | 0.00 |
| paid_full | Latest bill payment | the client paid the most recent bill in full and on time | `X6 (PAY_0) == -1` |  | 0.186 | -0.065 | 0.139 | 0.200 | 0.00 |
| low_limit | Credit limit | the client's credit limit is NT$50,000 or less | `X1 <= 50000` |  | 0.255 | +0.135 | 0.365 | 0.224 | 0.00 |
| high_util | Credit utilization | the current balance is 80% or more of the credit limit | `X12 / X1 >= 0.8` |  | 0.269 | +0.064 | 0.322 | 0.253 | 0.00 |
| tiny_payment | Size of last payment | the client's last payment covered less than 5% of the balance due | `X13 > 0 and X18 < 0.05 * X13` |  | 0.327 | +0.089 | 0.405 | 0.304 | 0.00 |
| no_payment | Recent payment activity | the client made no payment in either of the last two months while carrying a balance | `X18 == 0 and X19 == 0 and X12 > 0` |  | 0.040 | +0.071 | 0.066 | 0.033 | 0.00 |
| bal_growth | Balance trend | the outstanding balance grew by more than 20% over the last two months | `X12 > 1.2 * X14 and X12 > 0` |  | 0.283 | -0.109 | 0.192 | 0.310 | 0.00 |
| young | Client age | the client is under 25 years old | `X5 < 25` |  | 0.092 | +0.036 | 0.111 | 0.086 | 0.00 |
| edu_hs | Education | the client's highest education level is high school | `X3 == 3` |  | 0.164 | +0.021 | 0.179 | 0.160 | 0.00 |
| male | Sex | the client is male | `X2 == 1` |  | 0.394 | +0.041 | 0.431 | 0.383 | 0.00 |
| single | Marital status | the client is single | `X4 == 2` |  | 0.533 | -0.028 | 0.506 | 0.540 | 0.00 |
| old_bill | Statement six months ago | the bill six months ago exceeded NT$20,000 | `X17 > 20000` | yes | 0.447 | +0.010 | 0.456 | 0.444 | 0.00 |

Placebo candidate pool (corr_train): age_30s -0.0303, univ +0.0416, old_bill +0.0099, mid_limit -0.0307, age_40plus +0.0230; rule: |corr_train| < 0.02 (two smallest kept)

**cdc** (n=253680, train base rate 0.139)

| key | check name | finding text | threshold | placebo | prev | corr(f,Y) train | emp s1 | emp s0 | missing |
|---|---|---|---|---|---|---|---|---|---|
| high_bp | Blood-pressure history | the patient has been told they have high blood pressure | `HighBP == 1` |  | 0.428 | +0.262 | 0.750 | 0.376 | 0.00 |
| high_chol | Cholesterol history | the patient has been told they have high cholesterol | `HighChol == 1` |  | 0.424 | +0.198 | 0.668 | 0.385 | 0.00 |
| obese | Body-mass index | the patient's BMI is 30 or above | `BMI >= 30` |  | 0.347 | +0.198 | 0.582 | 0.309 | 0.00 |
| smoker | Smoking history | the patient has smoked at least 100 cigarettes in their life | `Smoker == 1` |  | 0.442 | +0.062 | 0.519 | 0.430 | 0.00 |
| stroke | Stroke history | the patient has had a stroke | `Stroke == 1` |  | 0.041 | +0.105 | 0.093 | 0.033 | 0.00 |
| heart | Heart history | the patient has coronary heart disease or a past heart attack | `HeartDiseaseorAttack == 1` |  | 0.094 | +0.176 | 0.221 | 0.073 | 0.00 |
| inactive | Physical activity | the patient did no physical activity outside work in the past 30 days | `PhysActivity == 0` |  | 0.243 | +0.117 | 0.368 | 0.223 | 0.00 |
| fair_poor | Self-rated health | the patient rates their general health as fair or poor | `GenHlth >= 4` |  | 0.172 | +0.251 | 0.408 | 0.134 | 0.00 |
| phys_days | Physical health days | the patient had poor physical health on 14 or more of the last 30 days | `PhysHlth >= 14` |  | 0.132 | +0.158 | 0.265 | 0.110 | 0.00 |
| diff_walk | Mobility | the patient has serious difficulty walking or climbing stairs | `DiffWalk == 1` |  | 0.168 | +0.216 | 0.368 | 0.135 | 0.00 |
| age60 | Age | the patient is 60 years or older | `Age >= 9 (60-64 and above)` |  | 0.483 | +0.161 | 0.682 | 0.450 | 0.00 |
| low_income | Household income | the household income is below $20,000 a year | `Income <= 3` |  | 0.148 | +0.122 | 0.256 | 0.130 | 0.00 |
| heavy_alcohol | Alcohol use | the patient is a heavy drinker (more than 14 drinks a week for men, 7 for women) | `HvyAlcoholConsump == 1` |  | 0.056 | -0.057 | 0.023 | 0.061 | 0.00 |
| no_veg | Vegetable intake | the patient does not eat vegetables at least once a day | `Veggies == 0` |  | 0.188 | +0.059 | 0.245 | 0.179 | 0.00 |
| coverage | Health-care coverage | the patient has some form of health-care coverage | `AnyHealthcare == 1` | yes | 0.952 | +0.017 | 0.961 | 0.950 | 0.00 |

Placebo candidate pool (corr_train): coverage +0.0170, cost_barrier +0.0290, male +0.0340, no_fruit +0.0418, mental_days +0.0669, college -0.1055; rule: |corr_train| < 0.02 (two smallest kept)

**german** (n=1000, train base rate 0.300)

| key | check name | finding text | threshold | placebo | prev | corr(f,Y) train | emp s1 | emp s0 | missing |
|---|---|---|---|---|---|---|---|---|---|
| chk_negative | Checking account balance | the applicant's checking account has a negative balance | `chk == A11 (< 0 DM)` |  | 0.270 | +0.265 | 0.450 | 0.193 | 0.00 |
| no_checking | Checking account existence | the applicant has no checking account with the bank | `chk == A14` |  | 0.397 | -0.323 | 0.156 | 0.500 | 0.00 |
| long_term | Loan duration | the requested loan term is 36 months or longer | `dur >= 36` |  | 0.172 | +0.194 | 0.283 | 0.124 | 0.00 |
| large_amount | Loan amount | the requested amount is 7,500 DM or more | `amount >= 7500` |  | 0.075 | +0.186 | 0.150 | 0.043 | 0.00 |
| past_delay | Credit history | the applicant had payment delays on past loans | `hist == A33` |  | 0.082 | -0.036 | 0.067 | 0.088 | 0.00 |
| critical_acct | Other existing credits | the applicant has a critical account or existing credits at other banks | `hist == A34` |  | 0.288 | -0.200 | 0.150 | 0.348 | 0.00 |
| low_savings | Savings | the applicant has less than 100 DM in savings | `savings == A61` |  | 0.602 | +0.139 | 0.706 | 0.557 | 0.00 |
| short_employ | Employment | the applicant has been employed for less than one year or is unemployed | `employ in {A71, A72}` |  | 0.245 | +0.109 | 0.317 | 0.214 | 0.00 |
| high_burden | Instalment burden | the instalment would fall in the highest burden category relative to disposable income | `instrate == 4` |  | 0.477 | +0.067 | 0.528 | 0.455 | 0.00 |
| young | Applicant age | the applicant is under 25 years old | `age < 25` |  | 0.167 | +0.078 | 0.211 | 0.148 | 0.00 |
| renter | Housing | the applicant rents their home | `housing == A151` |  | 0.182 | +0.144 | 0.267 | 0.145 | 0.00 |
| no_property | Property | the applicant owns no real estate, savings plan, life insurance or car | `property == A124` |  | 0.160 | +0.161 | 0.250 | 0.121 | 0.00 |
| other_plans | Other instalment plans | the applicant is paying off other instalment plans with banks or stores | `otherplans in {A141, A142}` |  | 0.188 | +0.075 | 0.233 | 0.169 | 0.00 |
| purpose_newcar | Loan purpose | the loan is for buying a new car | `purpose == A40` |  | 0.230 | +0.143 | 0.322 | 0.190 | 0.00 |
| long_residence | Residence duration | the applicant has lived at the current address for four years or more | `residence == 4` | yes | 0.415 | +0.002 | 0.417 | 0.414 | 0.00 |
| skilled_job | Occupation | the applicant works as a skilled employee or official | `job == A173` | yes | 0.630 | -0.003 | 0.628 | 0.631 | 0.00 |

Placebo candidate pool (corr_train): long_residence +0.0022, dependents +0.0180, skilled_job -0.0030, phone -0.0185, multi_credits -0.0820; rule: |corr_train| < 0.02 (two smallest kept)

**heart** (n=920, train base rate 0.553)

| key | check name | finding text | threshold | placebo | prev | corr(f,Y) train | emp s1 | emp s0 | missing |
|---|---|---|---|---|---|---|---|---|---|
| male | Sex | the patient is male | `sex == 1` |  | 0.797 | +0.262 | 0.892 | 0.680 | 0.00 |
| age55 | Age | the patient is 55 years or older | `age >= 55` |  | 0.502 | +0.277 | 0.626 | 0.348 | 0.00 |
| asympt | Chest-pain type | the patient reports no chest pain at all (asymptomatic presentation) | `cp == 4` |  | 0.514 | +0.533 | 0.754 | 0.219 | 0.00 |
| typical | Angina pattern | the chest pain is typical angina (exertional, relieved by rest) | `cp == 1` |  | 0.063 | -0.065 | 0.049 | 0.081 | 0.00 |
| high_bp | Resting blood pressure | the resting systolic blood pressure is 140 mm Hg or higher | `trestbps >= 140` |  | 0.368 | +0.115 | 0.419 | 0.308 | 0.07 |
| high_chol | Serum cholesterol | the serum cholesterol is 240 mg/dl or higher | `chol >= 240 (chol == 0 -> missing)` |  | 0.488 | +0.032 | 0.505 | 0.473 | 0.22 |
| high_fbs | Fasting blood sugar | the fasting blood sugar is above 120 mg/dl | `fbs == 1` |  | 0.177 | +0.103 | 0.215 | 0.136 | 0.10 |
| abn_ecg | Resting ECG | the resting ECG is abnormal (ST-T wave abnormality or left-ventricular hypertrophy) | `restecg in {1, 2}` |  | 0.403 | +0.071 | 0.434 | 0.364 | 0.00 |
| low_maxhr | Exercise heart rate | the maximum heart rate reached on the exercise test is below 120 beats per minute | `thalach < 120` |  | 0.230 | +0.238 | 0.320 | 0.120 | 0.06 |
| exang | Exercise symptoms | exercise induces angina | `exang == 1` |  | 0.373 | +0.443 | 0.567 | 0.137 | 0.06 |
| st_dep | Exercise ST depression | exercise causes ST-segment depression of 1 mm or more | `oldpeak >= 1` |  | 0.430 | +0.402 | 0.611 | 0.211 | 0.07 |
| flat_slope | ST slope | the ST segment at peak exercise is flat or downsloping | `slope in {2, 3}` |  | 0.645 | +0.348 | 0.767 | 0.417 | 0.34 |
| vessels | Fluoroscopy | fluoroscopy shows calcification in one or more major coronary vessels | `ca >= 1` |  | 0.423 | +0.378 | 0.613 | 0.240 | 0.66 |
| rev_defect | Thallium scan | the thallium stress scan shows a reversible perfusion defect | `thal == 7` |  | 0.433 | +0.412 | 0.609 | 0.196 | 0.53 |
| cath_firsthalf | Catheterization scheduling | the catheterization was booked for the first half of the calendar month | `cday <= 15` | yes | 0.457 | -0.045 | 0.438 | 0.483 | 0.34 |

Placebo candidate pool (corr_train): ekg_firsthalf -0.0301, cath_firsthalf -0.0446, ekg_winter +0.0373; rule: small-n fallback: min |corr_train| with p_train > 0.3 and |corr_all| < 0.02
## Reference model (scripts/fit_ref.py; LightGBM 4.6, CPU)
- **Y-model** P(Y=1 | any observed subset). LightGBM on the binary findings. TRAIN rows are replicated K times (taiwan 8,
  cdc 3, german/heart 60). Each replicate keeps a uniformly random number k in {0..F} of findings observed; the others
  are NaN (= unobserved). Early stopping uses masked CAL replicates. Calibration: isotonic on masked CAL for taiwan and cdc,
  Platt for german and heart (CAL has 200 records, where isotonic is step-y). Outputs are clipped to [1e-3, 1-1e-3].
- **f-models** P(f=1 | any observed subset of the other findings), one per finding, same masking (not recalibrated;
  masked-test ECE is reported in `models/<ds>/metrics.json`).

Out-of-sample on TEST (`models/<ds>/metrics.json`):

| ds | AUC full obs | Brier full (base-rate Brier) | ECE full | AUC masked (all k) | ECE masked | AUC with 1-2 obs | AUC with 3-5 | AUC with 6-8 |
|---|---|---|---|---|---|---|---|---|
| taiwan | 0.746 | 0.139 (0.172) | 0.016 | 0.692 | 0.004 | 0.616 | 0.667 | 0.696 |
| cdc | 0.808 | 0.100 (0.120) | 0.005 | 0.746 | 0.002 | 0.638 | 0.720 | 0.754 |
| german | 0.778 | 0.168 (0.210) | 0.054 | 0.694 | 0.016 | 0.524 | 0.601 | 0.733 |
| heart | 0.900 | 0.127 (0.247) | 0.062 | 0.817 | 0.034 | 0.700 | 0.784 | 0.859 |

Calibration curves (10 bins, masked test) are in `metrics.json["y_calibration_curve"]`. All four targets are clearly
above AUC 0.5 (cf. M7). For german, the 1-2-finding AUC is weak (0.52), so treat german as a small-data robustness set.

## Gold (scripts/gen.py): coherent preposterior on one joint per scenario
For evidence E (0-2 observed findings) and the 6 candidates C, we build one joint
`J[x_C, y] = p(x_C | E) * P(y | E, x_C)`:
- `p(x_C | E)` is the chain rule of masked f-models, averaged over 6 random chain orders.
- `P(y | E, x_C)` is the calibrated Y-model at each of the 64 leaves.

Every belief P(Y | E, x_S) and every predictive P(x_e = v | E, x_S) is a **marginal of J**. Therefore sum_v p(v) b_v = b
holds exactly, VOI >= 0, and the DP is internally consistent. Using the f-model for p(e|E) and the Y-model for the
posteriors separately would not be coherent: it creates spurious VOI near the threshold. `exact.posterior_direct` stores
the Y-model's own P(Y|E); `exact.coherence_gap` = posterior - posterior_direct. The mean |gap| is 0.012 for taiwan,
0.007 for cdc and 0.015 for german, but **0.075 for heart** (p95 0.18; small n plus heavy structural missingness). On heart
the joint root belief has the better test Brier (0.214 vs 0.227 direct).

The joint is stored per row (`joint`, 128 floats, index = cfg*2 + y, where bit k of cfg = value of check label k with
A = bit 0). The lead's harness can therefore recompute anything: sequential episodes, budget curves, or Q-values after
any history.

Definitions (TOL = 1e-9 everywhere; all probabilities clipped at 1e-3):
- `voi[L]` = sum_v p(v|E) max_a EU(a; b_v) - max_a EU(a; b). `eig[L]` is in bits. `net[L]` = voi - cost.
- `q[L]` = DP Q-value of acquiring L first.
- `dp_value` and `dp_first_action` come from the exact DP over 3^6 states. Stop is chosen whenever stop >= V - TOL.
  `dp_first_ties` holds the alternatives.
- `gold_choice` / `gold_ties`: myopic argmax net if > TOL, else "S".
- `stratum`: S1/S2/S3/other, using the same definitions as envs_synth (S2 additionally requires the DP to stop).
- `myopic_dp_disagree`: the myopic and DP first moves do not overlap.
- `nonmyopic_acquire`: myopic gold is S but the DP acquires. This is the non-myopic case of M5.

Independent brute-force DP re-check on stored joints: max error 1e-15 (`scripts/check.py`). min VOI = 0 and
dp >= myopic >= stop hold in every file.

## Scenario construction
- One scenario = one record. Natural files take one draw per sampled record.
- n_obs ~ {0,0,1,1,2} non-placebo findings, revealed with their actual values.
- 6 candidates come from the remaining non-missing findings. A placebo is included with prob 0.5 when one is available.
- The label order is random. `text` follows envs_synth (role, context, payoffs, base rate, already known, checks with
  costs, "(S) Stop and decide now").
- `text_parts.note_optional` ("How informative each check is, is not stated; judge it from your own knowledge.") is
  NOT in `text`. The lead decides whether to append it.
- `add_known(parts, check, value)` is in `scripts/gen.py`.
- Stored: `y_true` (realized outcome) and `checks[i].record_value` (the record's actual check value, for
  realized-utility episodes; never shown).
- Small sets (german, heart): every split record is used and then reused with fresh draws. `group_id = <ds>-<split>-r<rid>`;
  use it for clustered bootstrap.

## Files (data/)
| file | rows | unique records | content |
|---|---|---|---|
| taiwan_test / cdc_test .jsonl.gz | 2000 | 2000 | natural distribution (primary) |
| taiwan_dev / cdc_dev | 500 | 500 | natural |
| german_test / heart_test | 1000 | 200 / 184 | natural, records reused |
| german_dev / heart_dev | 300 | 200 / 184 | natural, records reused |
| *_test_strat / *_dev_strat | 1000/250 (large), 500/150 (small) | | stratified subset, quotas S1 .35 / S2 .25 / S3 .25 / other .15, drawn from natural + 8x extra draws on the same split's records (rejection only for this file) |
| {ds}_summary.json | | | strata counts, pool availability, diagnostics |
| checks_summary.json | | | sanity checks, other-stratum breakdown, stop rates |
| realized_utility_test.json | | | realized utility of DP / myopic / stop-now policies on actual records |

Row fields: id, track="tabular", domain, dataset, family (natural | pool), group_id, record_id, stratum, prior
(= train base rate), g, l, cost_regime, budget = 6, costs{A..F}, checks[{label, name, finding_text, key, cost, is_placebo,
emp_s1, emp_s0, record_value}], observed[{name, finding_text, key, value, emp_s1, emp_s0}], label_order, actions, states,
text_parts{context, payoff, base_rate, known, options, option_lines, stop_line, budget, note_optional}, text,
exact{posterior, posterior_direct, coherence_gap, eu_act, eu_pass, best_action, threshold, voi, eig, net, q, p_finding,
gold_choice, gold_ties, dp_value, dp_first_action, dp_first_ties, myopic_value, value_of_stopping, stratum,
myopic_dp_disagree, nonmyopic_acquire}, y_true, split, seed, joint.

## Strata and stop rates (natural TEST files)
| ds | S1 | S2 | S3 | other (stop / acquire) | gold stop | DP stop | myopic-DP disagree | non-myopic acquire |
|---|---|---|---|---|---|---|---|---|
| taiwan | 41 (2%) | 759 (38%) | 540 (27%) | 660 (571 / 89) | 0.68 | 0.65 | 3.7% | 3.3% |
| cdc | 219 (11%) | 233 (12%) | 243 (12%) | 1305 (1281 / 24) | 0.84 | 0.76 | 7.8% | 7.2% |
| german | 128 (13%) | 100 (10%) | 172 (17%) | 600 (574 / 26) | 0.76 | 0.69 | 7.9% | 7.4% |
| heart | 20 (2%) | 543 (54%) | 251 (25%) | 186 (175 / 11) | 0.73 | 0.61 | 12.8% | 11.4% |

"other" is dominated by myopic-stop cases outside S2 (max EIG <= 0.05 bits, or the DP acquires). All stratified files meet their quotas.

## Placebo verification
Marginal |corr(f,Y)| on TRAIN:
- taiwan old_bill: 0.010
- cdc coverage: 0.017 (prevalence 0.95)
- german long_residence: 0.002; skilled_job: 0.003
- heart cath_firsthalf: 0.045 train (p = 0.40), 0.018 on all records (fallback rule)

Conditional value under the reference model (natural TEST):

| ds | mean VOI placebo | frac placebo VOI > 0 | mean placebo VOI / max VOI | mean VOI non-placebo | frac non-placebo VOI > 0 |
|---|---|---|---|---|---|
| taiwan | 0.0007 | 2.7% | 0.006 | 0.037 | 29% |
| cdc | 0.0002 | 3.4% | 0.003 | 0.019 | 23% |
| german | 0.0003 | 2.7% | 0.003 | 0.028 | 29% |
| heart | 0.0068 | 9.3% | 0.035 | 0.056 | 25% |

Placebos are therefore close to zero-value but not exactly zero: the masked model finds small conditional dependence. The
`is_placebo` flag is semantic and empirical-marginal; the exact VOI is in `exact.voi`.

## Realized-utility validation on actual records (test, natural; mean ± s.e.)
| ds | stop now | DP policy under J | myopic policy | DP - stop | all 6 checks free (upper ref.) |
|---|---|---|---|---|---|
| taiwan | 0.710 | 0.735 | 0.735 | +0.025 ± 0.021 | 0.849 |
| cdc | 1.112 | 1.141 | 1.139 | +0.029 ± 0.015 | 1.204 |
| german | 0.394 | 0.428 | 0.434 | +0.034 ± 0.031 | 0.498 |
| heart | 0.124 | 0.222 | 0.213 | +0.098 ± 0.031 | 0.405 |

Following the gold policy with the records' true check values raises realized utility in all four datasets. The model
predicted gains (`dp_gain_over_stop_mean`) are 0.049, 0.022, 0.043 and 0.093 respectively.

## Caveats / not done
- The gold depends on the reference model (M7). Bootstrap refits for VOI-rank stability are **not** done yet; they are a
  possible follow-up (fit_ref takes about 1 min per dataset).
- Heart: coherence gap and f-model quality are weaker (n = 920, 34-66% missing for slope/ca/thal), and the placebo uses
  the fallback rule.
- German: small n, so treat it as a robustness set.
- The German credit data contain known counter-intuitive associations (e.g. "critical account" and "past delays" are
  associated with *lower* default). This is real-data "knowledge" that an LLM's priors may get wrong, which is informative
  for the paper but should be discussed.

## Reproduce
```
PY=python; S=envs_tab/scripts
$PY -I $S/prep.py taiwan cdc german heart   # clean/{ds}.csv.gz + clean/{ds}_meta.json
$PY -I $S/fit_ref.py <ds>                   # models/<ds>/
$PY -I $S/gen.py <ds>                       # data/
$PY -I $S/check.py ; $PY -I $S/check.py realized
```
(The scripts append the user site-packages path explicitly, since lightgbm and pandas live there and `-I` drops it.)
