"""Write paper/numbers.tex with \\newcommand macros computed from results/*.json, so the text never hard-codes numbers."""
import json, os, glob
import numpy as np
IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda n: json.load(open(f"{IS}/results/{n}.json")) if os.path.exists(f"{IS}/results/{n}.json") else {}
M = {}
NOTE = {}  # macro -> what is missing (macros computed on incomplete data expand to a red [TBD])
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from complete import (TXT, KNOW_LLMS, K4_LLMS, CTRL_LLMS, CTRL_BL_LLMS, REASON_CTRL, MS_LLMS, MS_DEV_LLMS, MS_QUAL_LLMS,
                      PROBE_LLMS, know_ok, know_missing, know_size, ctrl_ok)


def tbd(names, why):
    for n in names:
        if n in M or True:
            M[n] = TXT; NOTE[n] = why
DOMS = ["finance", "credit", "clinical", "legal"]
K = "dp_regret"

# controlled binary (common 300/domain subset with baselines)
B = J("bin_main")
bl_methods = ["bl_direct", "bl_cot_need", "bl_sc_cot", "bl_react", "bl_voi_prompt", "bl_voi_paper", "bl_finrisk_ask", "bl_bed_llm", "bl_pal", "think", "bl_never"]
div, best, never, direct, models_bin = [], [], [], [], []
_bin_missing = []
for tag in CTRL_BL_LLMS:
    row = B.get(tag, {})
    need = [m for m in bl_methods if m != "think" or tag in REASON_CTRL] + ["diva"]
    if not all(row.get(d, {}).get(m + "@sub", {}).get("n", 0) >= 300 for d in DOMS for m in need):
        _bin_missing.append(tag)
for tag, row in B.items():
    if tag not in CTRL_BL_LLMS:
        continue
    if not all(d in row and "diva@sub" in row[d] for d in DOMS):
        continue
    dv = np.mean([row[d]["diva@sub"][K] for d in DOMS])
    bls = {}
    for m in bl_methods:
        v = [row[d].get(m + "@sub", {}).get(K) for d in DOMS]
        if all(x is not None for x in v):
            bls[m] = np.mean(v)
    if len(bls) < 6:
        continue
    models_bin.append(tag); div.append(dv); best.append(min(v for k, v in bls.items() if k != "bl_never"))
    never.append(bls.get("bl_never", np.nan)); direct.append(bls.get("bl_direct", np.nan))
if div:
    M["NumModelsBin"] = len(models_bin)
    M["DivaBinReg"] = f"{100*np.mean(div):.2f}"
    M["BestBlBinReg"] = f"{100*np.mean(best):.2f}"
    M["RedBin"] = f"{100*(1-np.mean(div)/np.mean(best)):.0f}"  # best = best LLM baseline incl. native reasoning and the Dong et al. pipeline (never excluded below)
    M["RedBinMin"] = f"{100*min(1-d/b for d, b in zip(div, best)):.0f}"
    M["DirectBinReg"] = f"{100*np.nanmean(direct):.2f}"
    M["NeverBinReg"] = f"{100*np.nanmean(never):.2f}"
    M["DirectWorseThanNever"] = sum(d >= n * 0.95 for d, n in zip(direct, never))
M["NumModelsBin"] = len(CTRL_BL_LLMS)
if _bin_missing:
    tbd(["DivaBinReg", "BestBlBinReg", "RedBin", "RedBinMin", "DirectBinReg", "NeverBinReg", "DirectWorseThanNever"],
        "Table tab:main: first-300 baselines/DIVA incomplete for " + ",".join(_bin_missing))

# all models with DIVA on the full test (scaling etc.)
allm = [t for t, row in B.items() if all(d in row and "diva" in row[d] for d in DOMS)]
M["NumModels"] = len(set(allm))

# knowledge-required
for name in ("tab_main", "real_main"):
    R = J(name)
    wins_stop = wins_dir = wins_eig = cells = 0; ratios = []
    for tag, envs in R.items():
        for e, row in envs.items():
            if "diva" not in row or row["diva"]["n"] < 200:
                continue
            cells += 1
            d = row["diva"][K]; s = row["always_stop"][K]
            ratios.append(d / s if s > 0 else np.nan)
            wins_stop += d < s
            wins_dir += ("direct" in row and d < row["direct"][K])
            wins_eig += d < row["fac_eig"][K]
    key = "Tab" if name == "tab_main" else "Sec"
    M[f"{key}Cells"] = cells; M[f"{key}WinStop"] = wins_stop; M[f"{key}WinDirect"] = wins_dir; M[f"{key}WinEig"] = wins_eig
    if ratios:
        M[f"{key}Ratio"] = f"{np.nanmean(ratios):.2f}"

# multi-state fresh slice
R = J("ms_test_ms2")
c = w = 0; dv = []; dr = []; eg = []
for tag, envs in R.items():
    for e, row in envs.items():
        if "diva" in row and row["diva"]["n"] >= 300 and "direct" in row:
            c += 1; w += row["diva"][K] < min(row["direct"][K], row["fac_eig"][K], row["always_stop"][K])
            dv.append(row["diva"][K]); dr.append(row["direct"][K]); eg.append(row["fac_eig"][K])
M["MsCells"] = c; M["MsWins"] = w
if dv:
    M["MsDiva"] = f"{100*np.mean(dv):.2f}"; M["MsDirect"] = f"{100*np.mean(dr):.2f}"; M["MsEig"] = f"{100*np.mean(eg):.2f}"

# incoherence
I = J("analysis_incoherence")
if I:
    cs = [v["corr_update"] for v in I.values()]
    M["IncohCorrMin"] = f"{min(cs):.2f}".replace("-", "$-$"); M["IncohCorrMax"] = f"{max(cs):.2f}"
    # ratio of mean |exact| to mean |imagined| belief change, pooled over LLMs (equal n per LLM) = factor in Figure fig:incoh
    M["IncohShrink"] = f"{np.mean([v['mean_abs_update_exact'] for v in I.values()]) / max(np.mean([v['mean_abs_update_llm'] for v in I.values()]), 1e-6):.1f}"
    M["IncohTP"] = f"{100*np.mean([v['tp_violation'] for v in I.values()]):.1f}"
    M["NumIncoh"] = len(I)


# knowledge-required main table (results/know_main.json)
KM = J("know_main")
if KM:
    envs = ["tab:taiwan", "tab:german", "tab:cdc", "tab:heart", "sec:ret_1", "sec:ret_5", "sec:vol_5", "sec:vol_20"]
    cells = wins = 0; dratio = []; best = 0; best_nt = 0; uw = 0; vc = vw = 0
    for e in envs:
        recs = KM.get(e, {})
        TM = ["direct", "cot_need", "bed_llm", "voi_prompt", "pal", "fac_eig_cost", "never", "always_top", "diva", "diva_u"]
        trecs = [r for r in recs.values() if all(f"s300:{m}" in r for m in TM)]
        means = {m: np.mean([r[f"s300:{m}"][K] for r in trecs]) for m in TM} if trecs else {}
        if means and min(means, key=means.get) == "diva_u":
            best += 1
        nt = {k: v for k, v in means.items() if k != "fac_eig_cost"}
        if nt and min(nt, key=nt.get) == "diva_u":
            best_nt += 1
        for r in recs.values():
            if "full:diva_u" in r and "full:never" in r:
                cells += 1; wins += r["full:diva_u"][K] < r["full:never"][K]
                uw += r["full:diva_u"][K] < r["full:diva"][K]
                if e.startswith("sec:vol") and "full:direct" in r:
                    vc += 1; vw += r["full:diva_u"][K] <= r["full:direct"][K]
            if e.startswith("tab") and "full:direct" in r and "full:never" in r and r["full:never"][K] > 0:
                dratio.append(r["full:direct"][K] / r["full:never"][K])
    M["KnowCells"] = cells; M["KnowWinStop"] = wins; M["KnowDivaBest"] = best; M["KnowDivaBestNoTune"] = best_nt
    M["KnowUWinD"] = uw; M["VolCells"] = vc; M["VolWinDirect"] = vw
    wm = um = dm = 0
    for e in envs:
        for r in KM.get(e, {}).values():
            if "full:diva_u" in r and "full:never" in r:
                wm += r["full:diva_u"]["norm_regret"] < r["full:never"]["norm_regret"]
                um += r["full:diva_u"]["norm_regret"] < r["full:diva"]["norm_regret"]
                dm += r["full:diva"]["norm_regret"] < r["full:never"]["norm_regret"]
    M["KnowWinStopMyo"] = wm; M["KnowUWinDMyo"] = um; M["KnowDWinStopMyo"] = dm
    M["KnowDWinStop"] = sum(r["full:diva"][K] < r["full:never"][K] for e in envs for r in KM.get(e, {}).values() if "full:diva_u" in r and "full:never" in r)
    # K3 outcomes (PREREG_KNOW3) on the confirmation slices: records (all test rows, re-scored) and SEC rows >= 1000
    k3 = {"tab": [0, 0, 0], "sec": [0, 0, 0]}; k3c = [0, 0]
    for e in envs:
        for r in KM.get(e, {}).values():
            pre = "full" if e.startswith("tab") else "f1000"
            if all(f"{pre}:{m}" in r for m in ["diva", "diva_u", "never"]):
                t = k3["tab" if e.startswith("tab") else "sec"]
                t[0] += 1; t[1] += r[f"{pre}:diva_u"][K] < r[f"{pre}:diva"][K]; t[2] += r[f"{pre}:diva_u"][K] < r[f"{pre}:never"][K]
                if e.startswith("sec:vol") and f"{pre}:direct" in r:
                    k3c[0] += 1; k3c[1] += r[f"{pre}:diva_u"][K] <= r[f"{pre}:direct"][K]
    M["KthreeTabN"], M["KthreeTabA"], M["KthreeTabB"] = k3["tab"]
    M["KthreeSecN"], M["KthreeSecA"], M["KthreeSecB"] = k3["sec"]
    M["KthreeVolN"], M["KthreeVolC"] = k3c
    M["KnowDirectRatio"] = f"{np.median(dratio):.1f}" if dratio else "--"

LC = J("learning_curve")
if LC:
    def lcm(k):
        v = [r[k] / r["never"] for r in LC.values() if k in r and r["never"] > 0]
        return np.mean(v) if v else np.nan
    for k, name in [("llm:0", "LCLlmZero"), ("flat:0", "LCFlatZero"), ("flat:25", "LCFlatTwentyFive"), ("llm:25", "LCLlmTwentyFive"),
                    ("flat:50", "LCFlatFifty"), ("llm:50", "LCLlmFifty"), ("flat:100", "LCFlatHundred"), ("llm:100", "LCLlmHundred"), ("flat:-1", "LCFlatAll"), ("llm:1000", "LCLlmThousand"), ("generic:0", "LCGenZero"), ("generic:25", "LCGenTwentyFive")]:
        M[name] = f"{lcm(k):.2f}"
    M["LCCells"] = len(LC)
XC = J("extra_controls")
if XC:
    kn = XC.get("knowledge", {})
    d = [v["diva"][K] for v in kn.values()]; kf = [v["knowledge_free"][K] for v in kn.values()]
    if d:
        M["KFGap"] = f"{100*(np.mean(kf)/np.mean(d)-1):.0f}"
        M["KFCells"] = len(d)
        M["KFWorse"] = sum(1 for v in kn.values() if v["ci_vs_knowledge_free"][2] < 0)
    cc = XC.get("controlled", {})
    M["EIGTunedWins"] = sum(1 for v in cc.values() if v["diff_ci"][2] < 0); M["EIGTunedCells"] = len(cc)
    M["EIGTunedLoses"] = sum(1 for v in cc.values() if v["diff_ci"][1] > 0)
RZ = J("realized_tab")
if RZ:
    M["RealCells"] = len(RZ); M["RealPos"] = sum(1 for v in RZ.values() if v["ci"][1] > 0); M["RealNeg"] = sum(1 for v in RZ.values() if v["ci"][2] < 0)
    M["RealNegHeart"] = sum(1 for k, v in RZ.items() if k.endswith(":heart") and v["ci"][2] < 0)
MS2 = J("ms_test_ms2"); MSQ = J("ms_test_ms2q")
def _mtag(src, tag, m):  # same rule as the multi-state table: all four domains with >= 100 scenarios
    v = [src[tag][d][m][K] for d in ["finance", "credit", "clinical", "legal"] if tag in src and d in src[tag] and m in src[tag][d] and src[tag][d][m]["n"] >= 100]
    return None if len(v) < 4 else float(np.mean(v))
def _mavg(src, m, tags):
    v = [_mtag(src, t, m) for t in tags]; v = [x for x in v if x is not None]
    return f"{100*np.mean(v):.2f}" if v else "--"
if MS2:
    tags = [t for t in MS2 if _mtag(MS2, t, "diva") is not None]
    for mac, m in [("MsDiva", "diva"), ("MsDirect", "direct"), ("MsEig", "fac_eig"), ("MsEs", "fac_es"), ("MsElicited", "diva_elicited_payoff"), ("MsNever", "always_stop")]:
        M[mac] = _mavg(MS2, m, tags)
    M["MsTableLLMs"] = len(tags)
    TU = J("ms_tuned")
    if TU:
        tt = [t for t in tags if t in TU]
        M["MsEigTuned"] = _mavg(TU, "fac_eig_tuned", tt); M["MsEsTuned"] = _mavg(TU, "fac_es_tuned", tt); M["MsAlways"] = _mavg(TU, "always_top", tt)
        M["MsDivaDev"] = _mavg(MS2, "diva", tt)  # DIVA on the same eight development-tuned LLMs
        M["MsDevLLMs"] = len(tt)
        if sorted(tt) != sorted(MS_DEV_LLMS):
            tbd(["MsEigTuned", "MsEsTuned", "MsAlways", "MsDivaDev", "MsDevLLMs"], "tuned multi-state baselines incomplete")
if MSQ:
    qt = [t for t in MSQ if _mtag(MSQ, t, "diva") is not None and _mtag(MSQ, t, "direct") is not None and (not MS2 or _mtag(MS2, t, "diva") is not None)]
    M["MsQualDiva"] = _mavg(MSQ, "diva", qt); M["MsQualDirect"] = _mavg(MSQ, "direct", qt); M["MsQualLLMs"] = len(qt)
    if sorted(qt) != sorted(MS_QUAL_LLMS):
        tbd(["MsQualDiva", "MsQualDirect", "MsQualLLMs"], "verbal-belief multi-state runs incomplete")
TH = [B[t][d]["think"][K] for t in B for d in DOMS if "think" in B[t].get(d, {})] if B else []
if TH:
    M["ThinkReg"] = f"{100*np.mean(TH):.2f}"

TR = J("trap")
if TR:
    import collections as _c
    def aggtr(track):
        agg = _c.defaultdict(lambda: _c.defaultdict(float))
        for cell in TR[track].values():
            for m, st in cell.items():
                for k, v in st.items():
                    agg[m][k] += v
        return agg
    for track, P in [("controlled", "Ctl"), ("records", "Rec")]:
        A = aggtr(track)
        # the LLM baselines of Table tab:know (fact-check 10-12); controlled Direct = the Table-3 direct baseline
        llm = [m for m in (["direct"] if track == "controlled" else ["Direct"]) + ["cot_need", "sc_cot", "react", "finrisk_ask", "voi_prompt", "voi_paper", "bed_llm", "pal"] if m in A]
        def share(ms, types):
            num = sum(A[m]["reg_" + t] for m in ms for t in types); den = sum(A[m]["reg"] for m in ms)
            return 100 * num / den if den else float("nan")
        def rate(ms, key, den):
            return 100 * sum(A[m][key] for m in ms) / max(sum(A[m][den] for m in ms), 1)
        M[f"{P}LLMWaste"] = f"{share(llm, ['trap', 'placebo', 'overpriced']):.0f}"
        M[f"{P}LLMOverSuff"] = f"{rate(llm, 'over_suff', 'n_suff'):.0f}"
        M[f"{P}LLMTrap"] = f"{share(llm, ['trap']):.0f}"; M[f"{P}LLMPlacebo"] = f"{share(llm, ['placebo']):.0f}"; M[f"{P}LLMOverpriced"] = f"{share(llm, ['overpriced']):.0f}"
        if "voi_paper" in A:
            M[f"{P}VoiAskOverSuff"] = f"{rate(['voi_paper'], 'over_suff', 'n_suff'):.0f}"; M[f"{P}VoiAskPremature"] = f"{share(['voi_paper'], ['premature']):.0f}"
        if "DIVA-U" in A:
            M[f"{P}DivaUPlacebo"] = f"{share(['DIVA-U'], ['placebo']):.0f}"; M[f"{P}DivaPlacebo"] = f"{share(['DIVA'], ['placebo']):.0f}"
        M[f"{P}DivaOverSuff"] = f"{rate(['DIVA'], 'over_suff', 'n_suff'):.0f}"
        M[f"{P}DivaPremature"] = f"{share(['DIVA'], ['premature']):.0f}"
        if "Fac-EIG" in A:
            M[f"{P}EigTrapShare"] = f"{share(['Fac-EIG'], ['trap', 'overpriced']):.0f}"
            M[f"{P}EigOverSuff"] = f"{rate(['Fac-EIG'], 'over_suff', 'n_suff'):.0f}"
TF = J("transfer")
if TF:
    M["TransferSel"] = "/".join(sorted(set(str(tuple(v["selected"])) for v in TF.values())))
    M["TransferMean"] = f"{np.mean([v['test_selected'] for v in TF.values()]):.2f}"
    M["TransferPoint"] = f"{np.mean([round(v['test_point'], 2) for v in TF.values()]):.2f}"  # mean of the displayed values
    M["TransferGlobalU"] = f"{np.mean([v['test_global_u'] for v in TF.values()]):.2f}"
ST = J("analysis_strata")
if ST:
    def cst(r, m):
        v = [ST[t].get(f"cost:{r}|{m}", {}).get(K) for t in ST]; v = [x for x in v if x is not None]
        return f"{100*np.mean(v):.2f}" if v else "--"
    for r, nm in [("0.0", "Zero"), ("0.4", "High")]:
        for m, mm in [("diva", "Diva"), ("direct", "Direct"), ("fac_eig", "Eig")]:
            M[f"Cost{nm}{mm}"] = cst(r, m)
if XC:
    kn = XC.get("knowledge", {})
    if kn and all("diva_u" in v for v in kn.values()):
        M["KFGapU"] = f"{100*(np.mean([v['knowledge_free_u'][K] for v in kn.values()])/np.mean([v['diva_u'][K] for v in kn.values()])-1):.0f}"
if RZ and all("ci_u" in v for v in RZ.values()):
    M["RealPosU"] = sum(1 for v in RZ.values() if v["ci_u"][1] > 0); M["RealNegU"] = sum(1 for v in RZ.values() if v["ci_u"][2] < 0)
AG = J("alt_gold")
if AG:
    from scipy.stats import spearmanr as _sp
    xs, ys = [], []; uw = un = 0
    for ds, cells in AG.items():
        for c in cells.values():
            for m, v in c.items():
                xs.append(v["ref"]); ys.append(v["nb"])
            if "diva_u" in c and "never" in c:
                un += 1; uw += c["diva_u"]["nb"] < c["never"]["nb"]
    M["AltRho"] = f"{_sp(xs, ys).correlation:.2f}"; M["AltUCells"] = un; M["AltUWins"] = uw
EK4 = J("episodes_know_k4")
if EK4:
    envs8 = ["taiwan", "german", "cdc", "heart", "ret_1", "ret_5", "vol_5", "vol_20"]
    a = b_ = d_ = c_ = w_ = 0
    for e in envs8:
        from complete import K4_LLMS  # the 18 pre-registered LLMs of amendment 9 (same set as the CIs of Table tab:ci_main)
        cl = [v for t, v in EK4.get(e, {}).items() if t in K4_LLMS and all(m in v for m in ("never", "diva", "diva_dp", "diva_u"))]
        if not cl:
            continue
        mm = lambda k: np.mean([v[k]["regret_ref"] for v in cl])
        a += mm("diva_dp") < mm("never"); b_ += mm("diva_dp") < mm("diva")
        if e.startswith(("ret", "vol")):
            d_ += mm("diva_dp") < mm("diva_u")
        for v in cl:
            c_ += 1; w_ += v["diva_dp"]["regret_ref"] < v["never"]["regret_ref"]
    M["KfourA"] = a; M["KfourB"] = b_; M["KfourCells"] = c_; M["KfourC"] = w_; M["KfourD"] = d_
GD = J("gold_disagree")
if GD:
    M["GoldSameMin"] = f"{100*min(v['same_check_when_both_acquire'] for v in GD.values()):.0f}"; M["GoldSameMax"] = f"{100*max(v['same_check_when_both_acquire'] for v in GD.values()):.0f}"
    M["GoldStopDisMin"] = f"{100*min(v['myo_stop_dp_acq'] for v in GD.values()):.0f}"; M["GoldStopDisMax"] = f"{100*max(v['myo_stop_dp_acq'] for v in GD.values()):.0f}"
CIX = J("ci_tables")
if CIX and CIX.get("controlled"):
    M["CiBinSig"] = sum(1 for v in CIX["controlled"].values() if v["ci"][2] < 0); M["CiBinN"] = len(CIX["controlled"])
# ---------------- completeness gate (user rule: no unfinished numbers presented as final) ----------------
_km_tab = ["taiwan", "german", "cdc", "heart"]; _km_all = _km_tab + ["ret_1", "ret_5", "vol_5", "vol_20"]
def _miss(envs, tags=KNOW_LLMS, nslice=None):
    return sorted({t for e in envs for t in know_missing(e, tags, nslice)})
# legacy macros (old aggregation, unused in the text) and knowledge-required counts over all test scenarios
_m = _miss(_km_all)
if _m:
    tbd([k for k in M if k.startswith(("Tab", "Sec", "Know", "Vol"))], "knowledge-required specs incomplete (all test scenarios) for " + ",".join(_m))
    tbd(["KthreeTabN", "KthreeTabA", "KthreeTabB", "KthreeSecN", "KthreeSecA", "KthreeSecB", "KthreeVolN", "KthreeVolC"],
        "K3 counts need all 20 LLMs on all test scenarios; missing/partial: " + ",".join(_m))
_m = _miss(_km_tab, nslice=1000)
if _m or len(LC) != 80:
    tbd([k for k in M if k.startswith("LC")], "learning curve needs 20 LLMs x 4 datasets on the first 1,000 test scenarios; partial: " + ",".join(_m))
_m = _miss(_km_tab)
if _m or len(XC.get("knowledge", {})) != 80:
    tbd(["KFGap", "KFCells", "KFWorse", "KFGapU"], "knowledge-free control needs 20 LLMs x 4 datasets on all test scenarios; partial: " + ",".join(_m))
if _m or len(RZ) != 80:
    tbd(["RealCells", "RealPos", "RealNeg", "RealPosU", "RealNegU", "RealNegHeart"], "realized utility needs 20 LLMs x 4 datasets on all test scenarios; partial: " + ",".join(_m))
    tbd([k for k in M if k.startswith("Rec")], "trap anatomy on records needs complete runs; partial: " + ",".join(_m))
_mc = [t for t in CTRL_LLMS if not ctrl_ok(B, t)]
if _mc or set(XC.get("controlled", {})) != set(CTRL_LLMS):
    tbd(["EIGTunedWins", "EIGTunedCells", "EIGTunedLoses"], "tuned EIG vs DIVA needs all 20 LLMs on all 3,000 scenarios/domain; partial: "
        + ",".join(_mc) + "; no tuned EIG: " + ",".join(t for t in CTRL_LLMS if t not in XC.get("controlled", {})))
if _mc or _bin_missing:
    tbd([k for k in M if k.startswith("Ctl")], "trap anatomy (controlled) needs complete DIVA (3,000/domain) and baselines; partial: " + ",".join(sorted(set(_mc + _bin_missing))))
if _mc:
    tbd([k for k in M if k.startswith("Cost")] + ["ThinkReg"], "cost strata need all 20 LLMs on all 3,000 scenarios/domain; partial: " + ",".join(_mc))
_m = _miss(_km_all, nslice=400)
if _m:
    tbd(["TransferMean", "TransferPoint", "TransferGlobalU"], "transfer needs 20 LLMs on the first 400 test scenarios; partial: " + ",".join(_m))
_m = _miss(_km_tab, nslice=1000)
if _m or AG and not all(t in AG.get(d, {}) and "diva_u" in AG[d][t] for d in _km_tab for t in KNOW_LLMS):
    tbd(["AltRho", "AltUCells", "AltUWins"], "alternative reference needs 20 LLMs x 4 datasets on the first 1,000 scenarios; partial: " + ",".join(_m))
_m = _miss(_km_all, K4_LLMS, 1000)
if _m or not all(t in EK4.get(e, {}) for e in _km_all for t in K4_LLMS):
    tbd(["KfourA", "KfourB", "KfourC", "KfourCells", "KfourD"], "K4 episodes need the 18 non-70B LLMs on test scenarios 300-999; partial/missing: "
        + ",".join(sorted(set(_m) | {t for e in _km_all for t in K4_LLMS if t not in EK4.get(e, {})})))
if CIX and (set(CIX.get("controlled", {})) != set(CTRL_BL_LLMS) or _bin_missing):
    tbd(["CiBinSig", "CiBinN"], "controlled Delta CIs need every LLM of Table tab:main with complete baselines; incomplete: " + ",".join(_bin_missing))
if MS2 and sorted(t for t in MS2 if _mtag(MS2, t, "diva") is not None) != sorted(MS_LLMS):
    tbd([k for k in M if k.startswith("Ms") and "Qual" not in k], "multi-state test set incomplete")
if I and sorted(I) != sorted(PROBE_LLMS):
    tbd([k for k in M if k.startswith(("Incoh", "NumIncoh"))], "incoherence probe incomplete")

with open(f"{IS}/paper/numbers.tex", "w") as f:
    f.write("% Generated by scripts/make_numbers.py. Macros computed on incomplete data expand to a red [TBD];\n"
            "% the comment above each such macro says what is missing (intended LLM sets: scripts/complete.py).\n")
    for k, v in M.items():
        if k in NOTE:
            f.write(f"% TBD {k}: {NOTE[k]}\n")
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(M)
