# Knowledge-required environments: SEC filings

Buy-or-pass decisions at S&P 500 earnings announcements (8-K Item 2.02, 2025--2026 test period) for 1- and 5-day abnormal returns (Ret-1d, Ret-5d) and 5- and 20-day volatility (Vol-5d, Vol-20d). Checks are binned point-in-time findings from the filings; company names and dates are masked.

- `scripts/01_fetch_sec.py` to `scripts/05_validate.py`: Download from SEC EDGAR, build records, extract findings, build scenarios, and validate.
- `data/`: Records, bins, reference models, and scenarios; `voi/`: Exact decision values of the reference model.
- `eval_api.py`: Scoring API.

Raw EDGAR downloads are not included; set a contact User-Agent in `scripts/common.py` before fetching (SEC fair-access policy).
