"""Pre-specified binarization of real tabular records into natural-language findings + domain wording.

Thresholds were fixed from clinical / credit conventions and the column documentation BEFORE looking at any outcome
association. Placebo findings are chosen from a pre-declared CANDIDATE POOL (`placebo_pool`) by the empirical rule
|corr(f, Y)| < 0.02 on the TRAIN split (see prep.py); candidates failing the rule are dropped (never used as findings).
Each finding: key -> (check name, finding text, threshold documentation, function(df) -> float array with NaN=missing).
"""
import numpy as np

nan = np.nan


def _b(cond, missing=None):
    out = np.where(cond, 1.0, 0.0)
    if missing is not None:
        out = np.where(missing, nan, out)
    return out


# ------------------------------------------------------------------ Taiwan credit card default (UCI 350)
# X1 LIMIT_BAL (NT$), X2 sex (1 male, 2 female), X3 education (1 grad school, 2 university, 3 high school, 4 other),
# X4 marriage (1 married, 2 single, 3 other), X5 age, X6..X11 repayment status Sep..Apr (-2 no use, -1 paid duly,
# 0 revolving, k = k months delay), X12..X17 bill amounts Sep..Apr, X18..X23 payment amounts Sep..Apr. Y: default next month.
def _late_count(d):
    return sum((d[f"X{i}"] >= 1).astype(int) for i in range(6, 12))


TAIWAN = {
    "role": "a credit-risk analyst at a credit-card issuer",
    "context": "An existing credit-card client is up for a monthly credit-line review. You must decide whether to APPROVE "
               "continued use of the credit line for the coming month or DECLINE (suspend the line).",
    "act": "APPROVE", "pass": "DECLINE",
    "adverse": "the client defaults on next month's payment",
    "adverse_short": "default", "benign_short": "pay",
    "gain_text": "if you approve and the client pays, the issuer earns {g}; if you approve and the client defaults, "
                 "the issuer loses {l}. Declining yields 0 points either way.",
    "population": "credit-card clients of this issuer",
    "domain": "credit",
    "findings": {
        "late2_latest": ("Latest repayment status", "the client's most recent bill is two or more months overdue",
                         "X6 (PAY_0) >= 2", lambda d: _b(d.X6 >= 2)),
        "late_prev": ("Previous month's repayment status", "the client was at least one month late on the previous month's bill",
                      "X7 (PAY_2) >= 1", lambda d: _b(d.X7 >= 1)),
        "late3_of6": ("Six-month delinquency count", "the client was late on three or more of the last six monthly bills",
                      "#{X6..X11 >= 1} >= 3", lambda d: _b(_late_count(d) >= 3)),
        "paid_full": ("Latest bill payment", "the client paid the most recent bill in full and on time",
                      "X6 (PAY_0) == -1", lambda d: _b(d.X6 == -1)),
        "low_limit": ("Credit limit", "the client's credit limit is NT$50,000 or less",
                      "X1 <= 50000", lambda d: _b(d.X1 <= 50000)),
        "high_util": ("Credit utilization", "the current balance is 80% or more of the credit limit",
                      "X12 / X1 >= 0.8", lambda d: _b(d.X12 / d.X1 >= 0.8)),
        "tiny_payment": ("Size of last payment", "the client's last payment covered less than 5% of the balance due",
                         "X13 > 0 and X18 < 0.05 * X13", lambda d: _b((d.X13 > 0) & (d.X18 < 0.05 * d.X13))),
        "no_payment": ("Recent payment activity", "the client made no payment in either of the last two months while carrying a balance",
                       "X18 == 0 and X19 == 0 and X12 > 0", lambda d: _b((d.X18 == 0) & (d.X19 == 0) & (d.X12 > 0))),
        "bal_growth": ("Balance trend", "the outstanding balance grew by more than 20% over the last two months",
                       "X12 > 1.2 * X14 and X12 > 0", lambda d: _b((d.X12 > 1.2 * d.X14) & (d.X12 > 0))),
        "young": ("Client age", "the client is under 25 years old", "X5 < 25", lambda d: _b(d.X5 < 25)),
        "edu_hs": ("Education", "the client's highest education level is high school", "X3 == 3", lambda d: _b(d.X3 == 3)),
        "male": ("Sex", "the client is male", "X2 == 1", lambda d: _b(d.X2 == 1)),
        "single": ("Marital status", "the client is single", "X4 == 2", lambda d: _b(d.X4 == 2)),
    },
    "placebo_pool": {
        "age_30s": ("Age bracket", "the client is in their thirties", "30 <= X5 <= 39", lambda d: _b((d.X5 >= 30) & (d.X5 <= 39))),
        "univ": ("University degree", "the client holds a university degree (but not a graduate degree)", "X3 == 2",
                 lambda d: _b(d.X3 == 2)),
        "old_bill": ("Statement six months ago", "the bill six months ago exceeded NT$20,000", "X17 > 20000",
                     lambda d: _b(d.X17 > 20000)),
        "mid_limit": ("Credit-limit band", "the client's credit limit is between NT$100,000 and NT$200,000",
                      "100000 <= X1 <= 200000", lambda d: _b((d.X1 >= 100000) & (d.X1 <= 200000))),
        "age_40plus": ("Age 40 or older", "the client is 40 years old or older", "X5 >= 40", lambda d: _b(d.X5 >= 40)),
    },
}

# ------------------------------------------------------------------ Statlog German credit (UCI 144)
G_COLS = ["chk", "dur", "hist", "purpose", "amount", "savings", "employ", "instrate", "personal", "debtors",
          "residence", "property", "age", "otherplans", "housing", "ncredits", "job", "liable", "phone", "foreign", "target"]

GERMAN = {
    "role": "a loan officer at a retail bank",
    "context": "An individual has applied for a consumer instalment loan. You must decide whether to APPROVE the loan or DECLINE it.",
    "act": "APPROVE", "pass": "DECLINE",
    "adverse": "the applicant turns out to be a bad credit risk (does not repay as agreed)",
    "adverse_short": "bad risk", "benign_short": "good risk",
    "gain_text": "if you approve and the applicant repays as agreed, the bank earns {g}; if you approve and the applicant "
                 "turns out to be a bad credit risk, the bank loses {l}. Declining yields 0 points either way.",
    "population": "past applicants at this bank",
    "domain": "credit",
    "findings": {
        "chk_negative": ("Checking account balance", "the applicant's checking account has a negative balance",
                         "chk == A11 (< 0 DM)", lambda d: _b(d["chk"] == "A11")),
        "no_checking": ("Checking account existence", "the applicant has no checking account with the bank",
                        "chk == A14", lambda d: _b(d["chk"] == "A14")),
        "long_term": ("Loan duration", "the requested loan term is 36 months or longer", "dur >= 36", lambda d: _b(d["dur"] >= 36)),
        "large_amount": ("Loan amount", "the requested amount is 7,500 DM or more", "amount >= 7500",
                         lambda d: _b(d["amount"] >= 7500)),
        "past_delay": ("Credit history", "the applicant had payment delays on past loans", "hist == A33",
                       lambda d: _b(d["hist"] == "A33")),
        "critical_acct": ("Other existing credits", "the applicant has a critical account or existing credits at other banks",
                          "hist == A34", lambda d: _b(d["hist"] == "A34")),
        "low_savings": ("Savings", "the applicant has less than 100 DM in savings", "savings == A61",
                        lambda d: _b(d["savings"] == "A61")),
        "short_employ": ("Employment", "the applicant has been employed for less than one year or is unemployed",
                         "employ in {A71, A72}", lambda d: _b(d["employ"].isin(["A71", "A72"]))),
        "high_burden": ("Instalment burden", "the instalment would fall in the highest burden category relative to disposable income",
                        "instrate == 4", lambda d: _b(d["instrate"] == 4)),
        "young": ("Applicant age", "the applicant is under 25 years old", "age < 25", lambda d: _b(d["age"] < 25)),
        "renter": ("Housing", "the applicant rents their home", "housing == A151", lambda d: _b(d["housing"] == "A151")),
        "no_property": ("Property", "the applicant owns no real estate, savings plan, life insurance or car",
                        "property == A124", lambda d: _b(d["property"] == "A124")),
        "other_plans": ("Other instalment plans", "the applicant is paying off other instalment plans with banks or stores",
                        "otherplans in {A141, A142}", lambda d: _b(d["otherplans"].isin(["A141", "A142"]))),
        "purpose_newcar": ("Loan purpose", "the loan is for buying a new car", "purpose == A40", lambda d: _b(d["purpose"] == "A40")),
    },
    "placebo_pool": {
        "long_residence": ("Residence duration", "the applicant has lived at the current address for four years or more",
                           "residence == 4", lambda d: _b(d["residence"] == 4)),
        "dependents": ("Dependents", "the applicant supports two or more dependents", "liable >= 2", lambda d: _b(d["liable"] >= 2)),
        "skilled_job": ("Occupation", "the applicant works as a skilled employee or official", "job == A173",
                        lambda d: _b(d["job"] == "A173")),
        "phone": ("Telephone", "the applicant has a telephone registered in their name", "phone == A192",
                  lambda d: _b(d["phone"] == "A192")),
        "multi_credits": ("Existing credits at this bank", "the applicant already has two or more credits at this bank",
                          "ncredits >= 2", lambda d: _b(d["ncredits"] >= 2)),
    },
}

# ------------------------------------------------------------------ CDC Diabetes Health Indicators (UCI 891, BRFSS 2015)
CDC = {
    "role": "a primary-care physician reviewing a patient's annual health questionnaire",
    "context": "You must decide whether to SKIP a diagnostic referral for diabetes (blood-glucose / HbA1c testing) or REFER the patient for testing.",
    "act": "SKIP", "pass": "REFER",
    "adverse": "the patient has diabetes or prediabetes",
    "adverse_short": "have diabetes or prediabetes", "benign_short": "not have it",
    "gain_text": "if you skip the referral for a patient without diabetes or prediabetes, the outcome scores {g} "
                 "(avoided unnecessary testing); if you skip it for a patient who has diabetes or prediabetes, the outcome "
                 "scores minus {l}. Referring scores 0 points either way.",
    "population": "adults answering this national health survey",
    "domain": "clinical",
    "findings": {
        "high_bp": ("Blood-pressure history", "the patient has been told they have high blood pressure", "HighBP == 1",
                    lambda d: _b(d.HighBP == 1)),
        "high_chol": ("Cholesterol history", "the patient has been told they have high cholesterol", "HighChol == 1",
                      lambda d: _b(d.HighChol == 1)),
        "obese": ("Body-mass index", "the patient's BMI is 30 or above", "BMI >= 30", lambda d: _b(d.BMI >= 30)),
        "smoker": ("Smoking history", "the patient has smoked at least 100 cigarettes in their life", "Smoker == 1",
                   lambda d: _b(d.Smoker == 1)),
        "stroke": ("Stroke history", "the patient has had a stroke", "Stroke == 1", lambda d: _b(d.Stroke == 1)),
        "heart": ("Heart history", "the patient has coronary heart disease or a past heart attack", "HeartDiseaseorAttack == 1",
                  lambda d: _b(d.HeartDiseaseorAttack == 1)),
        "inactive": ("Physical activity", "the patient did no physical activity outside work in the past 30 days",
                     "PhysActivity == 0", lambda d: _b(d.PhysActivity == 0)),
        "fair_poor": ("Self-rated health", "the patient rates their general health as fair or poor", "GenHlth >= 4",
                      lambda d: _b(d.GenHlth >= 4)),
        "phys_days": ("Physical health days", "the patient had poor physical health on 14 or more of the last 30 days",
                      "PhysHlth >= 14", lambda d: _b(d.PhysHlth >= 14)),
        "diff_walk": ("Mobility", "the patient has serious difficulty walking or climbing stairs", "DiffWalk == 1",
                      lambda d: _b(d.DiffWalk == 1)),
        "age60": ("Age", "the patient is 60 years or older", "Age >= 9 (60-64 and above)", lambda d: _b(d.Age >= 9)),
        "low_income": ("Household income", "the household income is below $20,000 a year", "Income <= 3",
                       lambda d: _b(d.Income <= 3)),
        "heavy_alcohol": ("Alcohol use", "the patient is a heavy drinker (more than 14 drinks a week for men, 7 for women)",
                          "HvyAlcoholConsump == 1", lambda d: _b(d.HvyAlcoholConsump == 1)),
        "no_veg": ("Vegetable intake", "the patient does not eat vegetables at least once a day", "Veggies == 0",
                   lambda d: _b(d.Veggies == 0)),
    },
    "placebo_pool": {
        "coverage": ("Health-care coverage", "the patient has some form of health-care coverage", "AnyHealthcare == 1",
                     lambda d: _b(d.AnyHealthcare == 1)),
        "cost_barrier": ("Cost barrier", "the patient could not see a doctor in the past year because of cost", "NoDocbcCost == 1",
                         lambda d: _b(d.NoDocbcCost == 1)),
        "male": ("Sex", "the patient is male", "Sex == 1", lambda d: _b(d.Sex == 1)),
        "no_fruit": ("Fruit intake", "the patient does not eat fruit at least once a day", "Fruits == 0",
                     lambda d: _b(d.Fruits == 0)),
        "mental_days": ("Mental health days", "the patient had poor mental health on 14 or more of the last 30 days",
                        "MentHlth >= 14", lambda d: _b(d.MentHlth >= 14)),
        "college": ("Education", "the patient has a college degree (4 years or more)", "Education == 6",
                    lambda d: _b(d.Education == 6)),
    },
}

# ------------------------------------------------------------------ UCI Heart Disease (Cleveland, Hungary, Switzerland, VA)
H_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

HEART = {
    "role": "a cardiologist at an outpatient chest-pain clinic",
    "context": "A patient has been referred for evaluation of possible coronary artery disease. You must decide whether to "
               "DISCHARGE the patient without coronary angiography or REFER them for angiography.",
    "act": "DISCHARGE", "pass": "REFER",
    "adverse": "the patient has significant coronary artery disease (at least 50% narrowing of a major vessel)",
    "adverse_short": "have the disease", "benign_short": "not have it",
    "gain_text": "if you discharge a patient without significant disease, the outcome scores {g} (avoided an invasive "
                 "procedure); if you discharge a patient who has it, the outcome scores minus {l}. Referring scores 0 points either way.",
    "population": "patients referred to such clinics",
    "domain": "clinical",
    "findings": {
        "male": ("Sex", "the patient is male", "sex == 1", lambda d: _b(d.sex == 1, d.sex.isna())),
        "age55": ("Age", "the patient is 55 years or older", "age >= 55", lambda d: _b(d.age >= 55, d.age.isna())),
        "asympt": ("Chest-pain type", "the patient reports no chest pain at all (asymptomatic presentation)", "cp == 4",
                   lambda d: _b(d.cp == 4, d.cp.isna())),
        "typical": ("Angina pattern", "the chest pain is typical angina (exertional, relieved by rest)", "cp == 1",
                    lambda d: _b(d.cp == 1, d.cp.isna())),
        "high_bp": ("Resting blood pressure", "the resting systolic blood pressure is 140 mm Hg or higher", "trestbps >= 140",
                    lambda d: _b(d.trestbps >= 140, d.trestbps.isna() | (d.trestbps == 0))),
        "high_chol": ("Serum cholesterol", "the serum cholesterol is 240 mg/dl or higher", "chol >= 240 (chol == 0 -> missing)",
                      lambda d: _b(d.chol >= 240, d.chol.isna() | (d.chol == 0))),
        "high_fbs": ("Fasting blood sugar", "the fasting blood sugar is above 120 mg/dl", "fbs == 1",
                     lambda d: _b(d.fbs == 1, d.fbs.isna())),
        "abn_ecg": ("Resting ECG", "the resting ECG is abnormal (ST-T wave abnormality or left-ventricular hypertrophy)",
                    "restecg in {1, 2}", lambda d: _b(d.restecg >= 1, d.restecg.isna())),
        "low_maxhr": ("Exercise heart rate", "the maximum heart rate reached on the exercise test is below 120 beats per minute",
                      "thalach < 120", lambda d: _b(d.thalach < 120, d.thalach.isna())),
        "exang": ("Exercise symptoms", "exercise induces angina", "exang == 1", lambda d: _b(d.exang == 1, d.exang.isna())),
        "st_dep": ("Exercise ST depression", "exercise causes ST-segment depression of 1 mm or more", "oldpeak >= 1",
                   lambda d: _b(d.oldpeak >= 1, d.oldpeak.isna())),
        "flat_slope": ("ST slope", "the ST segment at peak exercise is flat or downsloping", "slope in {2, 3}",
                       lambda d: _b(d.slope >= 2, d.slope.isna())),
        "vessels": ("Fluoroscopy", "fluoroscopy shows calcification in one or more major coronary vessels", "ca >= 1",
                    lambda d: _b(d.ca >= 1, d.ca.isna())),
        "rev_defect": ("Thallium scan", "the thallium stress scan shows a reversible perfusion defect", "thal == 7",
                       lambda d: _b(d.thal == 7, d.thal.isna())),
    },
    "placebo_pool": {  # record-keeping attributes from the 76-attribute files (Hungary/Switzerland/VA; Cleveland raw file is corrupted -> missing)
        "ekg_firsthalf": ("Exercise-test scheduling", "the exercise ECG was recorded in the first half of the calendar month",
                          "ekgday <= 15", lambda d: _b(d.ekgday <= 15, d.ekgday.isna())),
        "cath_firsthalf": ("Catheterization scheduling", "the catheterization was booked for the first half of the calendar month",
                           "cday <= 15", lambda d: _b(d.cday <= 15, d.cday.isna())),
        "ekg_winter": ("Exercise-test season", "the exercise ECG was recorded in the winter months (December to February)",
                       "ekgmo in {12, 1, 2}", lambda d: _b(d.ekgmo.isin([12, 1, 2]), d.ekgmo.isna())),
    },
}

DATASETS = {"taiwan": TAIWAN, "german": GERMAN, "cdc": CDC, "heart": HEART}
