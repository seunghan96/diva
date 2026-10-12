#!/bin/bash
# Generate all splits for both tracks (CPU only). Sizes: dev 2000 / test 3000 per domain per track.
cd "$(dirname "$0")"
PY=python
W=${WORKERS:-12}
for d in finance credit clinical legal; do
  for s in dev test; do
    n=$([ $s = dev ] && echo 2000 || echo 3000)
    $PY gen_binary.py --domain $d --split $s --n $n --workers $W > logs/gen_binary_${d}_${s}.log 2>&1 &
  done
done
wait
for d in finance credit clinical legal; do
  for s in dev test; do
    n=$([ $s = dev ] && echo 2000 || echo 3000)
    $PY gen_ms.py --domain $d --split $s --n $n --workers $W > logs/gen_ms_${d}_${s}.log 2>&1 &
  done
done
wait
$PY make_aux.py > logs/make_aux.log 2>&1
