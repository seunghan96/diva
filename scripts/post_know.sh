#!/bin/bash
# Knowledge-track re-aggregation + tables + numbers + PDFs (does NOT touch Fig.1).
cd .; P=python
$P scripts/agg_know_main.py > logs/agg_know_main.log 2>&1
$P scripts/make_tables.py bin know2 ablation > logs/rb_tables.log 2>&1
$P scripts/make_tables_app.py >> logs/rb_tables.log 2>&1
$P scripts/make_numbers.py > logs/rb_numbers.log 2>&1
$P scripts/fit_tables.py
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main_full.tex > ../logs/rb_latex_full.log 2>&1; latexmk -pdf -interaction=nonstopmode main_8p.tex > ../logs/rb_latex.log 2>&1
echo POST_DONE $(date)
