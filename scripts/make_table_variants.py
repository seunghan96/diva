"""Main-text Table tab:variants (Analysis; user decision 2026-10-11): the strongest external baseline (VoI-Ask) vs DIVA, DIVA-U
and DIVA-DP, first-decision DP regret (Table tab:know cells, first 300 test scenarios) and episode regret (Table tab:episodes
cells), each averaged over the environments of a group and the same 20 LLMs. Reads results/variants_first.json
(make_tables.py know2) and results/variants_ep.json (episodes_all_table.py); TBD unless every cell is complete."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from complete import fmt_best, PH  # noqa: E402

IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = json.load(open(f"{IS}/results/variants_first.json"))
E = json.load(open(f"{IS}/results/variants_ep.json"))
ROWS = [("voi_ask", "voi_paper", "\\begin{tabular}[c]{@{}l@{}}VoI-Ask\\\\\\citep{dong2026value}\\end{tabular}"),
        ("diva", "diva", "\\diva{}"), ("diva_u", "diva_u", "\\diva{}-U"), ("diva_dp", "diva_dp", "\\diva{}-DP")]
cols = [[F.get(f, {}).get("know") for f, _, _ in ROWS], [F.get(f, {}).get("ctrl") for f, _, _ in ROWS],
        [E.get(e, {}).get("know") for _, e, _ in ROWS], [E.get(e, {}).get("ctrl") for _, e, _ in ROWS]]
cols = [fmt_best(c, scale=100) for c in cols]
from make_tables import merge_diva_rows  # noqa: E402
cols = merge_diva_rows([n for _, _, n in ROWS], cols, [1, 3])  # controlled: DIVA-U = DIVA, one centered value
lines = []
for i, (_, _, n) in enumerate(ROWS):
    lines.append(("\\rowcolor{oursrow} " if i else "") + n + " & " + " & ".join(c[i] for c in cols) + " \\\\")
    if i == 0:
        lines.append("\\midrule")
tex = ("\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{5pt}\n\\begin{tabular}{l|cc|cc}\n\\toprule\n"
       "\\multirow{2}{*}[-0.7ex]{Method} & \\multicolumn{2}{c|}{DP regret} & \\multicolumn{2}{c}{Episode regret} \\\\\n"
       "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n & Know. & Ctrl. & Know. & Ctrl. \\\\\n\\midrule\n"
       + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
       "\\caption{\\textbf{Myopic and planned acquisition.} Regret averaged over the knowledge-required (Know.) or controlled (Ctrl.) "
       "environments and 20 LLMs.}\n\\label{tab:variants}\n\\end{table}\n")
open(f"{IS}/paper/tables/variants.tex", "w").write(tex)
print("ok tab:variants", "TBD" if PH in tex else "complete")
