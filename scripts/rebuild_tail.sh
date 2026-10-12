#!/bin/bash
# Fast tail of rebuild_all.sh: tables, numbers, captions, checks and PDF from the existing aggregates (no re-aggregation).
cd .; P=python
$P scripts/ci_tables.py > logs/ci_tables.log 2>&1
$P scripts/episodes_all_table.py > logs/episodes_all.log 2>&1
$P scripts/fresh_table.py > logs/fresh.log 2>&1
$P scripts/make_tables_app.py >> logs/rb_tables.log 2>&1
$P scripts/sig_wins.py > logs/sig_wins.log 2>&1
$P scripts/make_tables.py bin know2 ablation abl_eig >> logs/rb_tables.log 2>&1
$P scripts/make_table_variants.py >> logs/rb_tables.log 2>&1; $P scripts/make_table_generic.py >> logs/rb_tables.log 2>&1  # Analysis tables tab:variants, tab:generic (need make_tables know2 + episodes_all_table)
$P scripts/prior_baselines.py > logs/rb_prior.log 2>&1  # F3: DeLLMa / UoT baselines (tab:prior)
$P scripts/agg_corr.py > logs/rb_corr.log 2>&1 && $P scripts/make_table_corr.py >> logs/rb_corr.log 2>&1  # F2: correlated evidence (tab:corr)
$P scripts/an_m1_decomp.py > logs/rb_m1.log 2>&1  # F1: selection vs stopping (tab:m1_decomp)
$P scripts/an_s5_ablation_know.py > logs/rb_s5.log 2>&1  # F1: ablation on knowledge-required envs (tab:abl_know)
$P scripts/an_s4_cost.py > logs/rb_s4.log 2>&1  # F1: cost of every method (tab:cost_all)
python scripts/an_m6_reference.py metrics > logs/rb_m6.log 2>&1 && python scripts/an_m6_reference.py rescore >> logs/rb_m6.log 2>&1 && python scripts/an_m6_reference.py tables >> logs/rb_m6.log 2>&1  # F1: reference validity (tab:ref_metrics, tab:alt_ref; alternative golds cached in results/m6_cache)
$P scripts/realized_big.py > logs/rb_f4_realized.log 2>&1; $P scripts/para_bl.py > logs/rb_f4_para.log 2>&1; $P scripts/tab_f4.py >> logs/rb_f4_para.log 2>&1  # F4: realized utility on 5,000 records (tab:realized_big), baseline paraphrases (tab:para_bl)
$P scripts/an_f8_agg.py > logs/f8_agg.log 2>&1; $P scripts/an_f8_tables.py > logs/f8_tables.log 2>&1  # F8 tables (an_f8_first.py / an_f8_divaus.py are long CPU runs, not re-run here)
$P scripts/make_numbers.py > logs/rb_numbers2.log 2>&1
(cd paper/figs/desiderata_candidates && pdflatex -interaction=nonstopmode d4.tex > /dev/null 2>&1 && cp d4.pdf ../fig_desiderata.pdf) && python -c "import sys; sys.path.insert(0, 'scripts'); from crop_figs import crop_pdf; crop_pdf('paper/figs/fig_desiderata.pdf')"  # Figure 1 reads its statistics from numbers.tex
python scripts/figs_all.py > logs/rb_figs.log 2>&1
python scripts/fig_walkthrough.py > logs/rb_walk.log 2>&1
$P scripts/fit_tables.py
$P scripts/apply_captions.py
python scripts/add_avg_rank.py
python scripts/add_avg_rank.py paper/tables/main_know_myo.tex
$P scripts/check_consistency.py > logs/check_consistency.log 2>&1
(cd paper && python3 ../scripts/check_refs.py main_full.tex > ../logs/check_refs.log 2>&1)
(cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main_full.tex > ../logs/rb_latex.log 2>&1)
(cd paper && python ../scripts/check_captions.py main_full.pdf > ../logs/check_captions.log 2>&1)
python scripts/check_valign.py paper/main_full.pdf > logs/check_valign.log 2>&1
echo TAIL_DONE $(date)
