#!/bin/bash
# Re-aggregate everything and rebuild tables, numbers, figures and the PDF.
cd .; P=python
$P scripts/extra_controls.py > logs/extra_controls_pre.log 2>&1
$P scripts/ctrl_kappa.py > logs/ctrl_kappa.log 2>&1
$P scripts/dp_big.py q8,oss,msm,g12,l8,g9,q25_7,m7,r1_7,r1_14,q25_14,phi4,q4b,q17b,q06b,g27,q14,q32,l70,q72 > logs/dp_big.log 2>&1
$P scripts/agg_bin.py q32,oss,msm,g12,q14,q8,l8,g9,q25_7,r1_14,m7,r1_7,q25_14,phi4,q4b,q17b,q06b,l70,q72,g27 > logs/rb_bin.log 2>&1 &
$P scripts/agg_other.py tab --lam 0.75 --form cond > logs/rb_tab.log 2>&1 &
$P scripts/agg_other.py realj --lam 1.0 > logs/rb_realj.log 2>&1 &
$P scripts/agg_other.py realj --lam 1.0 --minrow 1000 > logs/rb_realj1000.log 2>&1 &
$P scripts/agg_other.py ms --folder test_ms2 > logs/rb_ms2.log 2>&1 &
$P scripts/agg_other.py ms --folder test_ms2q > logs/rb_ms2q.log 2>&1 &
$P scripts/agg_episodes.py > logs/rb_ep.log 2>&1 &
$P scripts/analysis.py decomp incoherence > logs/rb_an.log 2>&1 &
wait
$P scripts/agg_other.py real --lam 0.75 > logs/rb_real.log 2>&1
$P scripts/extra_controls.py > logs/extra_controls.log 2>&1
$P scripts/kappa_tune.py > logs/kappa.log 2>&1
$P scripts/agg_know_main.py > logs/agg_know_main.log 2>&1
$P scripts/make_tables.py bin know2 ablation abl_eig > logs/rb_tables.log 2>&1
$P scripts/make_tables_app.py >> logs/rb_tables.log 2>&1
$P scripts/qual.py q8 > logs/rb_qual.log 2>&1; $P scripts/make_tables_app.py >> logs/rb_tables.log 2>&1
$P scripts/cost_table.py > logs/rb_cost.log 2>&1
$P scripts/realized_tab.py > logs/rb_realized.log 2>&1
python scripts/learning_curve.py > logs/learning_curve.log 2>&1
$P scripts/make_numbers.py > logs/rb_numbers.log 2>&1
$P scripts/trap_analysis.py > logs/trap.log 2>&1
$P scripts/analysis.py strata > logs/strata.log 2>&1
$P scripts/transfer.py > logs/transfer.log 2>&1
$P scripts/alt_gold.py > logs/alt_gold.log 2>&1
$P scripts/make_tables_app.py altgold >> logs/rb_tables.log 2>&1
EPK_PROC=48 $P scripts/episodes_know.py > logs/episodes_know.log 2>&1
EPK_SKIP=l70,q72 EPK_ROW0=300 EPK_NMAX=1000 EPK_POLS=never,diva,diva_u,diva_dp,direct1,ask1,beta,faceig,gen_dp,gen_u EPK_OUT=episodes_know_k4 EPK_PROC=64 $P scripts/episodes_know.py > logs/epk_k4.log 2>&1
$P scripts/ms_tuned.py > logs/ms_tuned.log 2>&1
$P scripts/ci_tables.py > logs/ci_tables.log 2>&1
$P scripts/const_strength.py > logs/const.log 2>&1
$P scripts/const_fair.py > logs/const_fair.log 2>&1
$P scripts/gold_disagree.py > logs/gold_disagree.log 2>&1
$P scripts/k4_table.py > logs/k4.log 2>&1
EPK_SKIP=none EPK_ROW0=0 EPK_NMAX=1000 EPK_POLS=never,diva,diva_u,diva_dp,direct1,ask1,always,beta,kappa,faceig,gen_dp,gen_u EPK_OUT=episodes_know_all EPK_PROC=48 $P scripts/episodes_know.py > logs/epk_all.log 2>&1
$P scripts/episodes_llm.py > logs/episodes_llm.log 2>&1
$P scripts/episodes_all_table.py > logs/episodes_all.log 2>&1
$P scripts/fresh_table.py > logs/fresh.log 2>&1
$P scripts/make_tables_app.py >> logs/rb_tables.log 2>&1
$P scripts/make_tables.py bin know2 ablation abl_eig >> logs/rb_tables.log 2>&1
$P scripts/voiask_know.py > logs/voiask_know.log 2>&1
$P scripts/make_numbers.py > logs/rb_numbers2.log 2>&1
python scripts/figs_all.py > logs/rb_figs.log 2>&1
python scripts/fig_walkthrough.py > logs/rb_walk.log 2>&1
python scripts/fig1.py > logs/rb_fig1.log 2>&1
[ -f paper/figs/fig1.pdf ] || cp paper/figs/fig1_cand1.pdf paper/figs/fig1.pdf
$P scripts/fit_tables.py
$P scripts/apply_captions.py
python scripts/add_avg_rank.py
python scripts/add_avg_rank.py paper/tables/main_know_myo.tex
$P scripts/check_consistency.py > logs/check_consistency.log 2>&1
(cd paper && python3 ../scripts/check_refs.py main_full.tex > ../logs/check_refs.log 2>&1)
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main_full.tex > ../logs/rb_latex.log 2>&1
(cd ./paper && python ../scripts/check_captions.py main_full.pdf > ../logs/check_captions.log 2>&1)
(cd . && python scripts/check_valign.py paper/main_full.pdf > logs/check_valign.log 2>&1)
echo REBUILD_DONE $(date)
