#!/bin/bash
# CPU: DIVA-family episodes for every (tag, domain) with >= 300 elicited test rows; skips done outputs.
cd .; P=python
for f in outputs/test_bin/facnum_*.jsonl.gz; do b=$(basename $f .jsonl.gz); b=${b#facnum_}; d=${b##*_}; t=${b%_*}
  O=outputs/episodes/ep_${t}_${d}.jsonl.gz; [ -s $O ] && continue
  n=$(zcat $f | wc -l); [ $n -lt 300 ] && continue
  $P scripts/run_episodes_fac.py envs_synth/data/binary/${d}_test.jsonl.gz $f $O --limit 300 --reps 4 --K 6 --oracle > logs/ep_${t}_${d}.log 2>&1 &
done; wait
