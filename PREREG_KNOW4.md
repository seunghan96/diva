# Amendment K4: sequential planning under the elicited specification (DIVA-DP) for full episodes on the
# knowledge-required tracks (before scoring DIVA-DP on ANY test data of these tracks)
Date 2026-10-09 (written before the confirmation run; the SHA-256 in PREREG_HASHES.txt is the binding record).
Motivation (disclosed): R6 showed that the first-decision real-data gains are weak under the myopic gold and that DIVA-U
over-acquires on SEC filings in full episodes. Dev-only diagnosis (scripts/dev_myopic_diag.py, dev_hybrid2.py,
dev_flat_switch.py, episodes_know.py with EPK_SPLIT=dev; results/*dev*.json): the myopic and DP golds agree on the
best action in 95-99% of the scenarios where both acquire, but disagree on how they score cheap checks whose value
arises in combination; the elicited prior is not the issue (stated vs elicited prior identical on dev); deferring the
check choice to the direct ranking helps only on volatility (not adopted); re-calibrating the final-action belief by
stronger shrinkage does not help (not adopted). Planning with exact DP under the SAME frozen specification
(DIVA-DP, already defined in the paper for controlled environments; no new parameters, lambda = 0.75, no sigma) had
lower dev episode regret (final action with the reference posterior) than DIVA in 8/8 environments and than never
asking in 8/8 environments (dev means over LLMs).
Primary metric: exact expected episode regret under the reference model with the final action taken with the
reference posterior (isolates acquisition quality); the policy's-own-belief variant is reported alongside.
Confirmation slice: test rows 300-999 of every knowledge-required environment (rows never used in any episode
analysis; DIVA-DP never scored on any test row of these tracks), all LLMs with frozen specifications on those rows
except 70B-class runs (incomplete).
Hypotheses: K4a DIVA-DP < never asking in >= 6 of 8 environments (mean over LLMs); K4b DIVA-DP < DIVA in >= 6 of 8
environments; K4c DIVA-DP < never asking in >= 2/3 of LLM-environment cells; K4d (secondary) DIVA-DP < DIVA-U on >= 3
of the 4 SEC targets. All outcomes are reported whether or not they hold; both first-decision golds remain reported.
