"""All-methods episode-regret tables on the knowledge-required environments (test rows 0-999), for the fixed LLM set
KNOW_LLMS (complete.py). DIVA family, tuned and reference policies: exact expectation (results/episodes_know_all.json);
prompting policies: multi-step episodes on two reference-model worlds per scenario (results/episodes_llm.json, unbiased
for the same expectation). A cell is a number only if every LLM of the fixed set has the method on all 1,000 scenarios
(prompting: every scenario with at least one sampled world); otherwise TBD. Writes paper/tables/episodes_all.tex and episodes_all_own.tex."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from complete import PH, KNOW_LLMS
from make_tables_app import _tab, ENVN

EXACT = [("never", "Never"), ("always_top", "Always ask"), ("ask1", r"Always ask $\rightarrow$ \diva{}"),
         ("kappa", r"\diva{}-$\kappa^\ddagger$"), ("beta", r"\diva{}-$\beta^\ddagger$")]
LLMM = [("direct", "Direct"), ("cot_need", "CoT"), ("sc_cot", "SC"), ("react", "ReAct"), ("finrisk_ask", "Ask-P"),
        ("voi_prompt", "VoI-P"), ("voi_paper", "VoI-Ask"), ("bed_llm", "BED-LLM"), ("pal", "PAL")]
OURS = [("diva", r"\diva{}"), ("diva_u", r"\diva{}-U"), ("diva_dp", r"\diva{}-DP")]
ENVS = ["taiwan", "german", "cdc", "heart", "ret_1", "ret_5", "vol_5", "vol_20"]
# EPA_ROWS=150 (default): every method on test rows 0-149 (prompting: s150 slice, 1-2 sampled worlds per scenario; exact policies:
# per-scenario exact values restricted to the same rows). EPA_ROWS=1000: rows 0-999 (needs the k4 prompting slice, z07).
NROWS = int(os.environ.get("EPA_ROWS", "150"))


_POS = {}


def _pos(e):
    """row position of each scenario in the *_epk file (rows 0-999 of the test split, the slice definition of episodes_llm)"""
    if e not in _POS:
        from evalx import read
        p = f"{IS}/envs_tab/data/{e}_epk.jsonl.gz" if e in ENVS[:4] else f"{IS}/envs_real/data/{e}_epk.jsonl.gz"
        _POS[e] = {s["id"]: k for k, s in enumerate(read(p))}
    return _POS[e]


def cell_exact(E, e, m, key):
    v = []
    for t in KNOW_LLMS:
        c = E.get(e, {}).get(t, {}).get(m)
        if not c or c.get("n", 0) < 1000:
            return None
        if NROWS >= 1000:
            v.append(c[key])
        else:
            vk = "vals_ref" if key == "regret_ref" else "vals"
            P = _pos(e)
            sel = [x for i, x in zip(c["ids"], c[vk]) if P.get(i, NROWS) < NROWS]
            if len(sel) < NROWS:
                return None
            v.append(float(np.mean(sel)))
    return 100 * float(np.mean(v))


def cell_llm(L, e, m, key):
    v = []
    for t in KNOW_LLMS:
        d = L.get(e, {}).get(t, {}).get(m, {})
        if NROWS < 1000:
            a = d.get(f"s{NROWS}")
            if not a or a["n"] < NROWS:
                return None
            v.append(a[key])
            continue
        a, b = d.get("s300"), d.get("k4")
        if not a or a["n"] < 300:
            return None
        if not b or b["n"] < 700:
            return None
        v.append((a[key] * a["n"] + b[key] * b["n"]) / (a["n"] + b["n"]))
    return 100 * float(np.mean(v))


def build(key, label, cap, keys=None):
    """rows = methods (canonical order), columns = the eight environments | Avg. (layout of Table tab:know)"""
    from complete import msort
    from make_tables import method_rows_table
    E = json.load(open(f"{IS}/results/episodes_know_all.json")) if os.path.exists(f"{IS}/results/episodes_know_all.json") else {}
    L = json.load(open(f"{IS}/results/episodes_llm.json")) if os.path.exists(f"{IS}/results/episodes_llm.json") else {}
    meths = msort(EXACT + LLMM + OURS, key=lambda x: x[1])  # canonical order by display name
    if keys is not None:
        meths = [x for x in meths if x[0] in keys]

    def val(m, e):
        x = cell_llm(L, e, m, key) if m in dict(LLMM) else cell_exact(E, e, m, key)
        return None if x is None else x / 100
    tex, _, _ = method_rows_table(meths, [(e, ENVN[e]) for e in ENVS], val, label, len(KNOW_LLMS),
                                  group=[(2, "Credit"), (2, "Clinical"), (4, "SEC filings (2025--2026)")], star=True, colsep="4.5pt")
    return tex.replace("\\caption{CAPTION}", "\\caption{" + cap + "}")


# main-text Table tab:episodes_know: exactly the methods of Table tab:know with real end-to-end episodes on test rows 0-149
# (Always ask has no stand-alone episode policy; its first decision followed by DIVA is reported as Always ask -> DIVA)
GENR = [("const_diva", r"Generic \diva{}"), ("const_diva_u", r"Generic \diva{}-U")]
GK = {"const_diva": "diva", "const_diva_u": "diva_u"}
MAIN_KEYS = [k for k, _ in LLMM] + ["never", "always_top", "diva", "diva_u"]
_UNUSED_GENERIC = ["const_diva", "const_diva_u"]


# controlled episode keys (results/episodes.json) of the main-table methods; ask1 has no controlled episode run (TBD)
CKEY = {"direct": "bl_direct", "cot_need": "bl_cot_need", "sc_cot": "bl_sc_cot", "react": "bl_react", "finrisk_ask": "bl_finrisk_ask",
        "voi_prompt": "bl_voi_prompt", "voi_paper": "bl_voi_paper", "bed_llm": "bl_bed_llm", "pal": "bl_pal", "never": "bl_never", "always_top": "always_top",
        "diva": "fiovoi@bl", "diva_u": "diva_u", "diva_dp": "fiovoi_dp@bl"}


def build_merged(key="regret_ref"):
    """main-text Table tab:episodes: the layout of Table tab:know, rows = the methods of Table
    tab:know with real end-to-end episodes, columns = knowledge-required environments (test rows 0-149) | controlled domains"""
    from complete import msort, KNOW_LLMS, EP_CTRL_LLMS, DOM_ORDER
    from make_tables import merged_method_table
    from make_tables_app import _ep_val
    E = json.load(open(f"{IS}/results/episodes_know_all.json")) if os.path.exists(f"{IS}/results/episodes_know_all.json") else {}
    L = json.load(open(f"{IS}/results/episodes_llm.json")) if os.path.exists(f"{IS}/results/episodes_llm.json") else {}
    C = json.load(open(f"{IS}/results/episodes.json")) if os.path.exists(f"{IS}/results/episodes.json") else {}
    ms = [x for x in msort(EXACT + LLMM + GENR + OURS, key=lambda x: x[1]) if x[0] in MAIN_KEYS]
    meths = [m for m, _ in ms]; names = [n for _, n in ms]
    llm = dict(LLMM)
    G = json.load(open(f"{IS}/results/generic_main.json")) if os.path.exists(f"{IS}/results/generic_main.json") else {}

    def kper(t, m, e):
        if m in GK:  # LLM-independent (scripts/generic_main.py)
            return G.get("know_ep", {}).get(GK[m], {}).get(e)
        if m in llm:
            a = L.get(e, {}).get(t, {}).get(m, {}).get(f"s{NROWS}")
            return a[key] if a and a["n"] >= NROWS else None
        c = E.get(e, {}).get(t, {}).get(m)
        if not c or c.get("n", 0) < 1000:
            return None
        P = _pos(e)
        sel = [x for i, x in zip(c["ids"], c["vals_ref" if key == "regret_ref" else "vals"]) if P.get(i, NROWS) < NROWS]
        return float(np.mean(sel)) if len(sel) >= NROWS else None

    def cper(t, m, d):
        if m in GK:
            return G.get("ctrl_ep", {}).get(GK[m], {}).get(d)
        return _ep_val(C, t, CKEY[m], d) if m in CKEY else None
    avg = lambda xs: float(np.mean(xs)) if all(x is not None for x in xs) else None
    kv = {(m, e): avg([kper(t, m, e) for t in KNOW_LLMS]) for m in meths for e in ENVS}
    cv = {(m, d): avg([cper(t, m, d) for t in EP_CTRL_LLMS]) for m in meths for d in DOM_ORDER}
    # episode group means of the strongest external baseline and our variants for Table tab:variants (scripts/make_table_variants.py)
    vm = ("voi_paper", "diva", "diva_u", "diva_dp")
    json.dump({m: {"know": avg([avg([kper(t, m, e) for t in KNOW_LLMS]) for e in ENVS]),
                   "ctrl": avg([avg([cper(t, m, d) for t in EP_CTRL_LLMS]) for d in DOM_ORDER])} for m in vm},
              open(f"{IS}/results/variants_ep.json", "w"), indent=1)
    # merged_method_table indexes knowledge environments by their Table-3 keys (tab:taiwan, sec:ret_1, ...)
    from make_tables import KNOW_ENVS
    ekey = {k: k.split(":")[1] for k, _ in KNOW_ENVS}
    kv = {(m, k): kv[(m, ekey[k])] for m in meths for k, _ in KNOW_ENVS}
    full = merged_method_table(meths, names, kv, cv, lambda t, m, k: kper(t, m, ekey[k]), cper, KNOW_LLMS, EP_CTRL_LLMS,
                               "tab:episodes_full", "episodes_full.tex")
    # main-text Table tab:episodes: compact single-column table, one column per environment group,
    # each the mean over that group's environments (TBD unless every environment cell is complete)
    from make_tables import _two_line, OURS_ROW, is_ours
    from complete import group_rows, fmt_best
    ka = [avg([kv[(m, k)] for k, _ in KNOW_ENVS]) for m in meths]
    ca = [avg([cv[(m, d)] for d in DOM_ORDER]) for m in meths]
    fk, fc = fmt_best(ka, scale=100), fmt_best(ca, scale=100)
    from make_tables import merge_diva_rows
    fk, fc = merge_diva_rows(list(names), [fk, fc], [1])  # controlled: DIVA-U = DIVA, one centered value
    fk, fc = merge_diva_rows(list(names), [fk, fc], [1], ("Generic \\diva{}", "Generic \\diva{}-U"))
    from make_tables import MCITE
    ONE = {"CoT", "SC", "ReAct", "Ask-P", "VoI-Ask", "PAL"}
    lab = lambda n: (n + " \\citep{" + MCITE[n] + "}") if n in ONE and n in MCITE else _two_line(n)
    lines = [(OURS_ROW if is_ours(n) else "") + lab(n) + " & " + fk[i] + " & " + fc[i] + " \\\\" for i, n in enumerate(names)]
    lines = group_rows(names, lines, 3)
    tex = ("\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{5pt}\n\\begin{tabular}{l|cc}\n\\toprule\n"
           # one header row: the two-line cells are [c] tabulars, so every header cell is centered on the row
           "\\begin{tabular}[c]{@{}l@{}}Method\\\\(Avg.\\ across " + str(len(KNOW_LLMS)) + " LLMs)\\end{tabular} & \\begin{tabular}[c]{@{}c@{}}Knowledge-\\\\required\\end{tabular} & Controlled \\\\\n"
           "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\caption{CAPTION}\n\\label{tab:episodes}\n\\end{table}\n")
    open(f"{IS}/paper/tables/episodes.tex", "w").write(tex)
    return full


if __name__ == "__main__":
    out = f"{IS}/paper/tables"
    print("tab:episodes pending cells:", build_merged())  # main text: merged knowledge-required | controlled episodes
    open(f"{out}/episodes_all.tex", "w").write(build("regret_ref", "tab:episodes_all", r"\textbf{Episode regret of every method} with the reference-posterior final action (test rows 0--149)."))
    open(f"{out}/episodes_all_own.tex", "w").write(build("regret", "tab:episodes_all_own", r"\textbf{Episode regret of every method} with the policy's own final action (test rows 0--149)."))
    print("ok")
