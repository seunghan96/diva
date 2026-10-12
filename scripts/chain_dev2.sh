#!/bin/bash
# wait for this model's pilot jobs, then run dev2 jobs sequentially
M=$1; T=$2; cd .; P=python
until grep -q "^done" logs/pilot_facnum_$T.log && grep -q "^done" logs/pilot_probe_$T.log && grep -q "^done" logs/pilot_facbin_$T.log; do sleep 30; done
$P scripts/run_factored.py $M data/dev_seed2.jsonl.gz outputs/dev2/facnum_$T.jsonl.gz --numeric > logs/dev2_facnum_$T.log 2>&1 &
$P scripts/run_probe.py $M data/dev_seed2.jsonl.gz outputs/dev2/probe_$T.jsonl.gz --pred > logs/dev2_probe_$T.log 2>&1 &
$P scripts/run_factored.py $M data/dev_seed2.jsonl.gz outputs/dev2/facbin_$T.jsonl.gz > logs/dev2_facbin_$T.log 2>&1
wait
