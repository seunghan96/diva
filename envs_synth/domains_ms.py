"""Multi-state domain specifications (factored hidden state, 3 actions, decision contexts).

Hidden state s = (level of factor 1, level of factor 2, level of factor 3); factors are a-priori independent
(the text states each factor's prior separately). Utility is additive over factors:
    U(a, s) = const[a] + sum_f W[context][f][a][level_f(s)]          (baseline action has U = 0)
A factor with W = 0 for every action in a context is decision-irrelevant there, so an item that only informs
about that factor has VOI exactly 0 while its EIG about the hidden state is > 0 (the designed "information trap").

Item outcome lists are ordered so that later outcomes are more typical of later factor levels; per-scenario
likelihoods are generated from this order (ordinal model) and stated in the text.
W entries are base values; the generator jitters them per scenario (x U(0.7, 1.3), rounded) and states them.
"""

FIN = {
    "name": "finance",
    "unit": "$k", "unit_word": "thousand dollars",
    "actions": ["BUY", "HOLD", "SELL"], "baseline": 1,
    "action_text": {"BUY": "BUY (open a long position)", "HOLD": "HOLD (take no position)",
                    "SELL": "SELL (open a short position)"},
    "factors": [
        {"key": "surprise", "name": "the revenue result versus consensus", "short": "revenue result",
         "levels": ["a miss", "in line", "a beat"], "prior": [0.25, 0.45, 0.30]},
        {"key": "guidance", "name": "the forward guidance", "short": "guidance",
         "levels": ["cut", "maintained", "raised"], "prior": [0.25, 0.50, 0.25]},
        {"key": "quality", "name": "the earnings quality", "short": "earnings quality",
         "levels": ["low (one-off or accrual-driven)", "high (recurring and cash-backed)"], "prior": [0.40, 0.60]},
    ],
    # return in % per factor level over the horizon (BUY gets +size*r, SELL gets -size*r); 0 = irrelevant at that horizon
    "contexts": [
        {"key": "1d", "horizon": "the next trading day",
         "why": "Over one day the price reacts to the headline print and the guidance; earnings quality is not priced that fast.",
         "ret": {"surprise": [-5.0, 0.0, 4.0], "guidance": [-3.0, 0.0, 2.0], "quality": [0.0, 0.0]}},
        {"key": "3m", "horizon": "the next three months",
         "why": "Over three months the print itself is already priced in; drift follows guidance and earnings quality.",
         "ret": {"surprise": [0.0, 0.0, 0.0], "guidance": [-6.0, 0.0, 4.0], "quality": [-4.0, 2.0]}},
        {"key": "12m", "horizon": "the next twelve months",
         "why": "Over a year, returns are driven by the durability of earnings and, to a lesser extent, guidance.",
         "ret": {"surprise": [0.0, 0.0, 0.0], "guidance": [-3.0, 0.0, 2.0], "quality": [-9.0, 5.0]}},
    ],
    "items": [
        ("Credit-card panel spending data", "surprise", ["weak", "in line", "strong"]),
        ("Supplier channel checks", "surprise", ["soft", "stable", "firm"]),
        ("Web-traffic and app-download tracker", "surprise", ["down", "up"]),
        ("Sell-side consensus revision tracker", "surprise", ["estimates drifting down", "estimates drifting up"]),
        ("Management commentary at an industry conference", "guidance", ["cautious", "neutral", "confident"]),
        ("Order-backlog disclosure in the latest 10-Q", "guidance", ["shrinking", "stable", "growing"]),
        ("Peer pre-announcements", "guidance", ["negative", "positive"]),
        ("Cash-flow versus earnings (accruals) analysis", "quality", ["accrual-heavy", "cash-backed"]),
        ("Revenue-recognition footnote review", "quality", ["aggressive", "conservative"]),
        ("Segment-mix disclosure", "quality", ["one-off driven", "mixed", "recurring driven"]),
        ("Deferred-revenue trend", "quality", ["falling", "rising"]),
    ],
    "placebos": [
        ("ESG rating update from a ratings vendor", ["downgrade", "upgrade"]),
        ("Social-media sentiment scrape (trending, high volume)", ["negative buzz", "positive buzz"]),
        ("8-K on the relocation of corporate headquarters", ["filed", "not filed"]),
        ("Proxy statement detail on board size", ["board shrank", "board grew"]),
        ("Exclusive TV interview with the chief executive (lifestyle segment)", ["subdued", "upbeat"]),
    ],
    "cost_scale": 4.0,      # typical item cost in $k
    "const": {"BUY": -3.0, "HOLD": 0.0, "SELL": -3.0},   # trading costs ($k)
}

CREDIT = {
    "name": "credit",
    "unit": "$k", "unit_word": "thousand dollars",
    "actions": ["APPROVE", "APPROVE_REDUCED", "DECLINE"], "baseline": 2,
    "action_text": {"APPROVE": "APPROVE the full amount", "APPROVE_REDUCED": "APPROVE_REDUCED (half the amount with tighter covenants)",
                    "DECLINE": "DECLINE the application"},
    "factors": [
        {"key": "capacity", "name": "the borrower's repayment capacity", "short": "repayment capacity",
         "levels": ["weak", "adequate", "strong"], "prior": [0.30, 0.45, 0.25]},
        {"key": "collateral", "name": "the collateral coverage", "short": "collateral coverage",
         "levels": ["thin", "solid"], "prior": [0.50, 0.50]},
        {"key": "integrity", "name": "the accuracy of the application", "short": "application accuracy",
         "levels": ["has an undisclosed problem", "clean"], "prior": [0.15, 0.85]},
    ],
    # value in $k relative to declining, per action and factor level
    "contexts": [
        {"key": "secured", "horizon": "a three-year secured term loan",
         "why": "For a secured loan, repayment capacity, collateral coverage and application accuracy all affect expected loss.",
         "W": {"APPROVE": {"capacity": [-80, 10, 40], "collateral": [-30, 15], "integrity": [-120, 0]},
               "APPROVE_REDUCED": {"capacity": [-30, 5, 15], "collateral": [-10, 5], "integrity": [-50, 0]}}},
        {"key": "unsecured", "horizon": "an unsecured revolving line of credit",
         "why": "The line is unsecured, so collateral plays no role in recovery.",
         "W": {"APPROVE": {"capacity": [-90, 10, 45], "collateral": [0, 0], "integrity": [-110, 0]},
               "APPROVE_REDUCED": {"capacity": [-35, 5, 18], "collateral": [0, 0], "integrity": [-45, 0]}}},
        {"key": "guaranteed", "horizon": "a government-guaranteed small-business loan (the guarantee covers credit losses unless the application was misstated)",
         "why": "The guarantee absorbs ordinary credit losses, so capacity and collateral barely matter; a misstated application voids the guarantee.",
         "W": {"APPROVE": {"capacity": [0, 0, 0], "collateral": [0, 0], "integrity": [-150, 0]},
               "APPROVE_REDUCED": {"capacity": [0, 0, 0], "collateral": [0, 0], "integrity": [-60, 0]}}},
    ],
    "items": [
        ("Twelve months of business bank statements", "capacity", ["frequent overdrafts", "steady balances", "growing balances"]),
        ("Owner's credit-bureau report", "capacity", ["recent delinquency", "clean file"]),
        ("Debt-service coverage from tax returns", "capacity", ["below 1.0x", "1.0x to 1.5x", "above 1.5x"]),
        ("Review of major customer contracts", "capacity", ["key contract expiring", "contracts renewed"]),
        ("Independent collateral appraisal", "collateral", ["below the loan amount", "near the loan amount", "well above the loan amount"]),
        ("Title and lien search", "collateral", ["prior liens found", "clear title"]),
        ("Equipment inspection", "collateral", ["worn or obsolete", "good condition"]),
        ("Litigation and judgment search", "integrity", ["open judgments found", "nothing found"]),
        ("Revenue verification against invoices", "integrity", ["mismatch", "match"]),
        ("Third-party reference calls", "integrity", ["concerns raised", "no concerns"]),
    ],
    "placebos": [
        ("Applicant's glossy marketing brochure", ["modest", "impressive"]),
        ("Feature article in an industry newsletter", ["critical", "glowing"]),
        ("Founder's social-media follower count (viral recent post)", ["small", "large"]),
        ("Office decor photos from the loan broker", ["plain", "lavish"]),
    ],
    "cost_scale": 3.0,
    "const": {"APPROVE": 0.0, "APPROVE_REDUCED": 0.0, "DECLINE": 0.0},
}

CLIN = {
    "name": "clinical",
    "unit": "outcome points", "unit_word": "outcome points (higher is better)",
    "actions": ["SUPPORTIVE", "ANTIBIOTICS", "ADMIT"], "baseline": 0,
    "action_text": {"SUPPORTIVE": "SUPPORTIVE care and discharge with safety-net advice",
                    "ANTIBIOTICS": "ANTIBIOTICS (oral, outpatient)", "ADMIT": "ADMIT for intravenous therapy and monitoring"},
    "factors": [
        {"key": "etiology", "name": "the cause of the respiratory illness", "short": "cause",
         "levels": ["viral", "non-infectious", "bacterial"], "prior": [0.50, 0.15, 0.35]},
        {"key": "severity", "name": "the severity of the illness", "short": "severity",
         "levels": ["mild", "severe"], "prior": [0.75, 0.25]},
        {"key": "drugrisk", "name": "the patient's risk of a serious drug interaction", "short": "interaction risk",
         "levels": ["high", "low"], "prior": [0.20, 0.80]},
    ],
    # outcome points relative to supportive care
    "contexts": [
        {"key": "amoxicillin", "horizon": "a community clinic where the first-line antibiotic is amoxicillin",
         "why": "Amoxicillin has no relevant interactions with the patient's medicines, so interaction risk does not matter.",
         "W": {"ANTIBIOTICS": {"etiology": [-2, -2, 10], "severity": [0, -6], "drugrisk": [0, 0]},
               "ADMIT": {"etiology": [-3, -1, 12], "severity": [-5, 15], "drugrisk": [0, 0]}}},
        {"key": "fluoroquinolone", "horizon": "a clinic where, due to the patient's penicillin allergy, the only antibiotic option is a fluoroquinolone",
         "why": "Fluoroquinolones interact with some of the patient's medicines; in hospital the interaction can be monitored.",
         "W": {"ANTIBIOTICS": {"etiology": [-2, -2, 10], "severity": [0, -6], "drugrisk": [-10, 0]},
               "ADMIT": {"etiology": [-3, -1, 12], "severity": [-5, 15], "drugrisk": [-2, 0]}}},
        {"key": "remote", "horizon": "a remote clinic with no hospital transfer before tomorrow (admission is not possible today)",
         "why": "Admission is not possible today, so only the choice between antibiotics and supportive care matters; severity does not change that choice.",
         "W": {"ANTIBIOTICS": {"etiology": [-2, -2, 10], "severity": [0, 0], "drugrisk": [0, 0]},
               "ADMIT": {"etiology": [-30, -30, -30], "severity": [0, 0], "drugrisk": [0, 0]}}},
    ],
    "items": [
        ("Procalcitonin test", "etiology", ["low", "intermediate", "high"]),
        ("C-reactive protein test", "etiology", ["low", "moderate", "high"]),
        ("Chest X-ray", "etiology", ["clear", "patchy changes", "lobar consolidation"]),
        ("Rapid viral PCR panel", "etiology", ["virus detected", "no virus detected"]),
        ("Pulse oximetry", "severity", ["95% or higher", "below 95%"]),
        ("Vital-signs severity score (CURB-65)", "severity", ["0 to 1", "2", "3 or more"]),
        ("Serum lactate", "severity", ["normal", "elevated"]),
        ("ECG for QT interval", "drugrisk", ["prolonged QT", "normal QT"]),
        ("Pharmacist medication reconciliation", "drugrisk", ["interacting medicine found", "no interacting medicine"]),
    ],
    "placebos": [
        ("Dental history review", ["recent dental work", "no recent dental work"]),
        ("Tetanus vaccination record", ["out of date", "up to date"]),
        ("Family history of migraine", ["present", "absent"]),
        ("Detailed travel history (holiday photos shared by the patient)", ["travelled abroad", "no travel"]),
    ],
    "cost_scale": 0.6,
    "const": {"SUPPORTIVE": 0.0, "ANTIBIOTICS": 0.0, "ADMIT": 0.0},
}

LEGAL = {
    "name": "legal",
    "unit": "$k", "unit_word": "thousand dollars (for the client)",
    "actions": ["TRIAL", "MEDIATE", "SETTLE"], "baseline": 2,
    "action_text": {"TRIAL": "TRIAL (litigate to judgment)", "MEDIATE": "MEDIATE (non-binding mediation first)",
                    "SETTLE": "SETTLE now at the offered amount"},
    "factors": [
        {"key": "merits", "name": "the strength of the plaintiff's liability case", "short": "liability case",
         "levels": ["strong", "contested", "weak"], "prior": [0.30, 0.40, 0.30]},
        {"key": "damages", "name": "the provable damages", "short": "provable damages",
         "levels": ["high", "low"], "prior": [0.40, 0.60]},
        {"key": "records", "name": "the client's document-retention record", "short": "document retention",
         "levels": ["gaps (risk of an adverse-inference ruling)", "complete"], "prior": [0.20, 0.80]},
    ],
    # value in $k for the client relative to settling now
    "contexts": [
        {"key": "uncapped", "horizon": "a dispute with no contractual cap on damages",
         "why": "With no cap, trial exposure depends on liability, damages and any adverse-inference ruling.",
         "W": {"TRIAL": {"merits": [-300, 40, 250], "damages": [-150, 60], "records": [-120, 0]},
               "MEDIATE": {"merits": [-60, 30, 90], "damages": [-40, 20], "records": [-20, 0]}}},
        {"key": "capped", "horizon": "a dispute where an enforceable limitation-of-liability clause caps damages at the settlement amount",
         "why": "The cap makes the size of provable damages irrelevant to the comparison with settling.",
         "W": {"TRIAL": {"merits": [-150, 40, 250], "damages": [0, 0], "records": [-100, 0]},
               "MEDIATE": {"merits": [-40, 30, 90], "damages": [0, 0], "records": [-20, 0]}}},
        {"key": "bench", "horizon": "a bench trial before a judge who has ruled that retention gaps will not support an adverse inference",
         "why": "The court has already ruled that document-retention gaps will not affect the outcome.",
         "W": {"TRIAL": {"merits": [-300, 40, 250], "damages": [-150, 60], "records": [0, 0]},
               "MEDIATE": {"merits": [-60, 30, 90], "damages": [-40, 20], "records": [0, 0]}}},
    ],
    "items": [
        ("Interview of the key witness", "merits", ["supports the plaintiff", "ambiguous", "supports the client"]),
        ("Targeted email discovery", "merits", ["damaging emails", "nothing damaging"]),
        ("Precedent search in the jurisdiction", "merits", ["favours the plaintiff", "favours the client"]),
        ("Independent damages expert report", "damages", ["above the offer", "below the offer"]),
        ("Review of the plaintiff's financial statements", "damages", ["large documented losses", "small documented losses"]),
        ("Forensic IT review of document retention", "records", ["gaps found", "no gaps"]),
        ("Records-custodian interviews", "records", ["inconsistent accounts", "consistent accounts"]),
    ],
    "placebos": [
        ("Count of press articles about the case (trending on legal news)", ["many", "few"]),
        ("Opposing counsel's law-school ranking", ["top-ranked", "lower-ranked"]),
        ("Courthouse renovation schedule", ["renovation planned", "no renovation"]),
        ("Mock-jury poll on the client's logo", ["dislike", "like"]),
    ],
    "cost_scale": 8.0,
    "const": {"TRIAL": -40.0, "MEDIATE": -10.0, "SETTLE": 0.0},
}

MS_DOMAINS = {"finance": FIN, "credit": CREDIT, "clinical": CLIN, "legal": LEGAL}
