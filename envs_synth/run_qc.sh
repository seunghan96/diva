#!/bin/bash
# QC round-trip over all splits (resumable). Priority: pilot, test1000, then full test, then dev.
cd "$(dirname "$0")"
PY=python
M=${QC_MODEL:-Qwen/Qwen3-32B-AWQ}
for pat in pilot test1000 test dev; do
  for t in binary ms; do
    $PY qc_roundtrip.py --model $M --conc ${CONC:-48} --files data/$t/*_${pat}.jsonl.gz
  done
done
