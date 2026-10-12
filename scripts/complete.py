"""Completeness rules for every table cell and text macro.

A number is printed only when it is computed on the COMPLETE intended set: every LLM of the intended LLM set below and
every scenario of the stated slice, for every compared method. Otherwise tables print PH (gray TBD) and numbers.tex
macros expand to TXT (red [TBD]) with a comment saying what is missing.

Intended LLM sets (from the paper text, captions and the job files in jobs/*.tsv):
  KNOW_LLMS      knowledge-required environments (20 tags)
  K4_LLMS        pre-registered knowledge episodes (KNOW_LLMS minus the 70B-class LLMs, PREREG_KNOW4)
  REASON_KNOW    native reasoning on knowledge-required environments (jobs/g4_know_think.tsv)
  CTRL_LLMS      controlled environments, DIVA and direct policy (jobs/fac_test_bin + m_test_dir_bin + q_scaling)
  CTRL_BL_LLMS   controlled LLM baselines, Table tab:main (jobs/c_bl_test.tsv + g3_g27_bl.tsv)
  REASON_CTRL    native reasoning on controlled environments (jobs/c2_think.tsv)
  EP_CTRL_LLMS   controlled episodes with LLM baselines (jobs/d_bl_ep.tsv)
  MS_LLMS / MS_DEV_LLMS / MS_QUAL_LLMS   multi-state test / development-tuned / verbal-belief sets (b2 / b_ms_dev / b3)
  PROBE_LLMS / ABL_LLMS   incoherence probe / ablation (a2_probe_raw / a1_abl)
  K4SPEC_LLMS    sampled specifications (y_test_tab_k4 / y_test_real_k4)
  PARA_LLMS      paraphrases (c4_para + g5_para_more)
"""
import json, os
IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PH = "\\textcolor{gray}{TBD}"
TXT = "\\textcolor{red}{[TBD]}"

KNOW_LLMS = ["q06b", "q17b", "q4b", "q8", "q14", "q32", "q25_7", "q25_14", "q72", "l8", "l70", "g9", "g27", "g12",
             "m7", "msm", "phi4", "oss", "r1_7", "r1_14"]
K4_LLMS = [t for t in KNOW_LLMS if t not in ("l70", "q72")]
REASON_KNOW = ["q06b", "q17b", "q4b", "q8", "q14", "q32", "oss", "r1_7", "r1_14"]  # F4 10-12: Qwen3-0.6B added (supports thinking; outputs valid)
CTRL_LLMS = list(KNOW_LLMS)
CTRL_BL_LLMS = list(KNOW_LLMS)
REASON_CTRL = list(REASON_KNOW)  # same reasoning-capable LLMs as knowledge envs
EP_CTRL_LLMS = list(KNOW_LLMS)
MS_LLMS = list(KNOW_LLMS)
MS_DEV_LLMS = list(KNOW_LLMS)
MS_QUAL_LLMS = list(KNOW_LLMS)
PROBE_LLMS = list(KNOW_LLMS)
ABL_LLMS = list(KNOW_LLMS)
K4SPEC_LLMS = list(KNOW_LLMS)
PARA_LLMS = list(KNOW_LLMS)

KENVS = ["tab:taiwan", "tab:german", "tab:cdc", "tab:heart", "sec:ret_1", "sec:ret_5", "sec:vol_5", "sec:vol_20"]
CDOMS = ["finance", "credit", "clinical", "legal"]
CTRL_N = 3000  # test scenarios per controlled domain

_KM = None


def _km():
    global _KM
    if _KM is None:
        p = f"{IS}/results/know_main.json"
        _KM = json.load(open(p)) if os.path.exists(p) else {}
    return _KM


def kenv(e):
    """'taiwan' -> 'tab:taiwan'"""
    return e if ":" in e else next((k for k in KENVS if k.endswith(":" + e)), e)


def know_size(e):
    """number of test scenarios of a knowledge-required environment (largest elicitation coverage over LLMs)"""
    recs = _km().get(kenv(e), {})
    return max((r.get("full:diva", {}).get("n", 0) for r in recs.values()), default=0)


def know_n(e, tag):
    return _km().get(kenv(e), {}).get(tag, {}).get("full:diva", {}).get("n", 0)


def know_ok(e, tags=KNOW_LLMS, nslice=None):
    """every LLM of `tags` has the elicited specification on the first `nslice` (default: all) test scenarios of env e"""
    size = know_size(e)
    need = size if nslice is None else min(nslice, size)
    return size > 0 and all(know_n(e, t) >= need for t in tags)


def know_missing(e, tags=KNOW_LLMS, nslice=None):
    size = know_size(e); need = size if nslice is None else min(nslice, size)
    return [t for t in tags if know_n(e, t) < need]


def ctrl_n(B, tag, m="diva"):
    return [B.get(tag, {}).get(d, {}).get(m, {}).get("n", 0) for d in CDOMS]


def ctrl_ok(B, tag, m="diva"):
    return all(n >= CTRL_N for n in ctrl_n(B, tag, m))


# ---------------------------------------------------------------------------------------------------------------------
# Canonical orders and display names: every table, figure and list uses these orders and names.
# Generators sort with msort()/lsort()/esort(); scripts/check_consistency.py asserts them on paper/tables/*.tex.
# ---------------------------------------------------------------------------------------------------------------------
DAG = "$^\\ddagger$"
# canonical METHOD order (display names, LaTeX). core list with extra methods slotted in:
#   prompting (Direct ... Reason.), extra LLM controls, EIG/ES policies, reference policies, hybrids, generic, tuned DIVA, DIVA.
METHOD_GROUPS = [
    ("Prompting", ["Direct", "CoT", "SC", "ReAct", "Reason.", "Salience", "Uncertainty"]),
    ("Decision-aware prompting", ["Ask-P", "VoI-P", "VoI-Ask"]),
    ("Information gain", ["BED-LLM", "EIG (beliefs)"]),
    ("Program-aided", ["PAL", "VoI-Tool"]),  # VoI-Tool: the LLM fills in a provided VOI calculator (additional analyses)
    ("Reference", ["Never", "Always ask", "Direct $\\rightarrow$ \\diva{}", "Always ask $\\rightarrow$ \\diva{}", "Random", "Threshold heur.", "Supervised ref.", "Knowledge-free \\diva{}", "Knowledge-free \\diva{}-U",
                   "\\diva{}-$\\kappa^\\ddagger$", "\\diva{}-$\\beta^\\ddagger$"]),
    # LLM-free specification (every check the same strength
    ("LLM-free", ["Generic DP", "Generic \\diva{}", "Generic \\diva{}-U"]),
    # variants of DIVA with its criterion replaced (not baselines
    ("Ablation", ["\\diva{} w/ EIG", "\\diva{} w/ EIG" + DAG, "\\diva{} w/ ES", "\\diva{} w/ ES" + DAG]),
    ("Ours", ["\\diva{} (elic.)", "\\diva{}", "\\diva{}-U", "\\diva{}-DP"]),
]
METHOD_ORDER = [m for _, ms in METHOD_GROUPS for m in ms]
METHOD_GROUP = {m: g for g, ms in METHOD_GROUPS for m in ms}
# DIVA variants whose decision-value criterion is replaced by EIG/ES: shown only in ablation tables, never as baselines
ABLATION_KEYS = {"fac_eig", "fac_eig@bl", "fac_eig_cost", "faceig", "fac_eig_tuned", "fac_es", "fac_es_tuned"}


def group_rows(names, rows, ncols):
    """insert an italic group-header row (\\multicolumn{1}{l|}{\\textit{Group}} & & ... with continuous vertical rules) before each method category of a
    method-row table and a \\midrule between categories; names[i] is the display name of rows[i] (canonical order)."""
    out, cur = [], None
    for n, r in zip(names, rows):
        g = METHOD_GROUP.get(n)
        if g != cur:
            if cur is not None or out:
                out.append("\\midrule")
            # the group label spans every column (no column rules in this row), left-aligned
            out.append("\\multicolumn{" + str(ncols) + "}{l}{\\textit{" + (g or "Other") + "}} \\\\")
            cur = g
        out.append(r)
    return out
# method keys used by the generators -> canonical display name
MNAME = {
    "direct": "Direct", "bl_direct": "Direct", "Direct": "Direct",
    "cot_need": "CoT", "bl_cot_need": "CoT", "sc": "SC", "sc_cot": "SC", "bl_sc_cot": "SC", "react": "ReAct", "bl_react": "ReAct",
    "ask_p": "Ask-P", "finrisk_ask": "Ask-P", "bl_finrisk_ask": "Ask-P", "voi_prompt": "VoI-P", "bl_voi_prompt": "VoI-P",
    "voi_ask": "VoI-Ask", "voi_paper": "VoI-Ask", "bl_voi_paper": "VoI-Ask", "bed_llm": "BED-LLM", "bl_bed_llm": "BED-LLM",
    "pal": "PAL", "bl_pal": "PAL", "think": "Reason.",
    "bl_salience": "Salience", "bl_uncertainty": "Uncertainty", "bl_eig": "EIG (beliefs)", "bl_random": "Random",
    "fac_eig": "\\diva{} w/ EIG", "fac_eig@bl": "\\diva{} w/ EIG", "fac_eig_cost": "\\diva{} w/ EIG" + DAG, "faceig": "\\diva{} w/ EIG" + DAG,
    "fac_eig_tuned": "\\diva{} w/ EIG" + DAG, "fac_es": "\\diva{} w/ ES", "fac_es_tuned": "\\diva{} w/ ES" + DAG, "heuristic": "Threshold heur.",
    "never": "Never", "always_stop": "Never", "bl_never": "Never", "always_top": "Always ask", "always": "Always ask",  # acquire every check, then decide (= Always ask)
    # direct1 / ask1: hybrid episode policies, first decision from Direct / Always ask, then DIVA (named honestly, 2026-10-11)
    "direct1": "Direct $\\rightarrow$ \\diva{}", "ask1": "Always ask $\\rightarrow$ \\diva{}", "gen_dp": "Generic DP", "const_diva": "Generic \\diva{}",
    "const_diva_u": "Generic \\diva{}-U", "emp_nb": "Supervised ref.",
    "kappa": "\\diva{}-$\\kappa^\\ddagger$", "diva_kappa": "\\diva{}-$\\kappa^\\ddagger$",
    "beta": "\\diva{}-$\\beta^\\ddagger$", "diva_beta": "\\diva{}-$\\beta^\\ddagger$",
    "knowledge_free": "Knowledge-free \\diva{}", "diva_elicited_payoff": "\\diva{} (elic.)", "diva": "\\diva{}", "fiovoi@bl": "\\diva{}",
    "knowledge_free_u": "Knowledge-free \\diva{}-U", "diva_u": "\\diva{}-U", "diva_dp": "\\diva{}-DP", "fiovoi_dp@bl": "\\diva{}-DP",
}
# citations shown under the display name in the main tables (second line of the cell)
MCITE = {"Ask-P": "zhong2026finriskatlas", "CoT": "wei2022chain", "SC": "wang2023selfconsistency", "ReAct": "yao2023react", "VoI-Ask": "dong2026value",
         "BED-LLM": "choudhury2026bedllm", "PAL": "gao2023pal"}


def mname(k):
    return MNAME.get(k, k)


def _mbase(name):
    """canonical display name of a (possibly qualified) method header, e.g. '\\diva{} (DP)' -> '\\diva{}'"""
    import re as _re
    n = _re.sub(r"\\citep\{[^}]*\}", "", name)
    _pb = _re.match(r"\s*(\\cellcolor\{[^}]*\})?\\parbox\[[a-z]\]\{[^}]*\}\{\\centering\s*(.*)\}\s*$", n, _re.S)
    if _pb:  # cited method header: \parbox[c]{w}{\centering Name\\\citep{..}} (citation already removed)
        n = _pb.group(2)
    n = _re.sub(r"\\cellcolor\{[^}]*\}|\\textbf\{|\\shortstack\[[a-z]\]\{|\\begin\{tabular\}\[[a-z]\]\{@\{\}[a-z]@\{\}\}|\\end\{tabular\}", "", n)
    n = n.replace("\\\\", " ").replace("}", "} ").strip()
    n = _re.sub(r"\s+", " ", n).replace("} ", "}").replace("- ", "-").strip()  # two-line headers, e.g. BED-\\\\LLM
    for m in sorted(METHOD_ORDER, key=len, reverse=True):
        if n == m:
            return m
    return None


def mrank(k):
    """position of a method key or display name in METHOD_ORDER"""
    n = MNAME.get(k, k)
    return METHOD_ORDER.index(n) if n in METHOD_ORDER else len(METHOD_ORDER)


def msort(items, key=lambda x: x[0]):
    """sort (method_key, ...) items into the canonical method order (stable for unknown keys)"""
    return sorted(items, key=lambda x: mrank(key(x)))


# canonical LLM order: family-grouped (Table 7 layout C), ascending size within a family
LLM_FAMILIES = [("Qwen3", ["q06b", "q17b", "q4b", "q8", "q14", "q32"]), ("Qwen2.5", ["q25_7", "q25_14", "q72"]),
                ("DeepSeek-R1", ["r1_7", "r1_14"]), ("Llama", ["l8", "l70"]), ("Gemma", ["g9", "g12", "g27"]),
                ("Mistral", ["m7", "msm"]), ("Phi", ["phi4"]), ("gpt-oss", ["oss"])]
LLM_ORDER = [t for _, ts in LLM_FAMILIES for t in ts]
NAMES = {"q06b": "Qwen3-0.6B", "q17b": "Qwen3-1.7B", "q4b": "Qwen3-4B", "q8": "Qwen3-8B", "q14": "Qwen3-14B", "q32": "Qwen3-32B",
         "q25_7": "Qwen2.5-7B", "q25_14": "Qwen2.5-14B", "q72": "Qwen2.5-72B", "r1_7": "R1-Qwen-7B", "r1_14": "R1-Qwen-14B",
         "l8": "Llama-3.1-8B", "l70": "Llama-3.3-70B", "g9": "Gemma-2-9B", "g12": "Gemma-3-12B", "g27": "Gemma-2-27B",
         "m7": "Mistral-7B", "msm": "Mistral-Small-24B", "phi4": "Phi-4", "oss": "gpt-oss-20B"}
NAMES = {t: NAMES[t] for t in LLM_ORDER}  # dict order = canonical order
FAMILY_OF = {t: f for f, ts in LLM_FAMILIES for t in ts}


def lsort(tags):
    """canonical LLM order (unknown tags last)"""
    return sorted(tags, key=lambda t: LLM_ORDER.index(t) if t in LLM_ORDER else len(LLM_ORDER))


# canonical environment / domain order and names
ENV_ORDER = ["taiwan", "german", "cdc", "heart", "ret_1", "ret_5", "vol_5", "vol_20"]
DOM_ORDER = ["finance", "credit", "clinical", "legal"]
ENVN = {"taiwan": "Taiwan", "german": "German", "cdc": "CDC", "heart": "Heart", "ret_1": "Ret-1d", "ret_5": "Ret-5d",
        "vol_5": "Vol-5d", "vol_20": "Vol-20d", "finance": "Finance", "credit": "Credit", "clinical": "Clinical", "legal": "Legal"}


def esort(envs):
    o = ENV_ORDER + DOM_ORDER
    b = lambda e: e.split(":")[-1]
    return sorted(envs, key=lambda e: o.index(b(e)) if b(e) in o else len(o))


def fmt_best(vals, nd=2, scale=1.0):
    """format a list of lower-is-better values (None -> TBD): best bold, second best underlined"""
    xs = sorted(set(round(v * scale, nd) for v in vals if v is not None))
    out = []
    for v in vals:
        if v is None:
            out.append(PH); continue
        r = round(v * scale, nd); s = f"{r:.{nd}f}"
        if xs and r == xs[0]:
            s = "\\textbf{" + s + "}"
        elif len(xs) > 1 and r == xs[1]:
            s = "\\underline{" + s + "}"
        out.append(s)
    return out


# family-grouped LLM rows for every per-LLM table:
# column 1 = Family (bold, multirow, vertically centered), column 2 = Size, a \cmidrule between families.
SIZE_OF = {"q06b": "0.6B", "q17b": "1.7B", "q4b": "4B", "q8": "8B", "q14": "14B", "q32": "32B", "q25_7": "7B", "q25_14": "14B",
           "q72": "72B", "r1_7": "7B", "r1_14": "14B", "l8": "8B", "l70": "70B", "g9": "9B", "g12": "12B", "g27": "27B",
           "m7": "7B", "msm": "24B", "phi4": "14B", "oss": "20B"}
FAMILY_NOTE = "Gemma: Gemma-2/3, Llama: Llama-3.1/3.3, Mistral 24B: Mistral-Small, DeepSeek-R1: R1-Distill-Qwen."
FAM_HEAD = "Family & Size"
FAM_LABEL = {}


def family_rows(tags, cells, ncols, lead=None):
    """rows for the LLMs `tags` (canonical order, grouped by family). cells(tag) -> list of the remaining cells (strings).
    ncols = total number of columns of the tabular (for the \\cmidrule). lead = optional list of cells placed before Family
    (e.g. the environment of a stacked table); only its first row is filled."""
    out = []
    groups = [(f, [t for t in ts if t in tags]) for f, ts in LLM_FAMILIES]
    groups = [(f, ts) for f, ts in groups if ts]
    k0 = 0 if lead is None else len(lead)
    for gi, (f, ts) in enumerate(groups):
        if gi:
            out.append("\\cmidrule(lr){" + f"{k0 + 1}-{ncols}" + "}")
        for i, t in enumerate(ts):
            fl = FAM_LABEL.get(f, f) if len(ts) > 1 else f
            fam = ("\\multirow{" + str(len(ts)) + "}{*}{\\textbf{" + fl + "}}") if i == 0 else ""
            pre = "" if lead is None else (" & ".join(lead) if (gi == 0 and i == 0) else " & ".join([""] * len(lead))) + " & "
            out.append(pre + fam + " & " + SIZE_OF[t] + " & " + " & ".join(cells(t)) + " \\\\")
    return out
