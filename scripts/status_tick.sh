#!/bin/bash
# every 2 h: append compact progress to ../STATUS.md; when every jobs/*.tsv is complete, run the final rebuild_all once.
cd .
while true; do
  p=$(python3 scripts/pending_all.py 2>/dev/null | tr "\n" "," )
  r=$(pgrep -f scripts/rebuild_all.sh >/dev/null && echo "rebuild running" || echo "last rebuild: $(tail -1 logs/rebuild.log)")
  al=$(tail -3 logs/allocator.log | tr '\n' ' ' | cut -c1-300)
  echo "- $(date '+%m-%d %H:%M') [auto] pending: ${p:-none}. ${r}. allocator: ${al}" >> ../STATUS.md
  if [ -z "$p" ] && [ ! -f logs/final_rebuild.done ]; then
    while pgrep -f scripts/rebuild_all.sh >/dev/null; do sleep 120; done
    echo "- $(date '+%m-%d %H:%M') [auto] ALL REQUIRED JOBS COMPLETE -> final rebuild_all started" >> ../STATUS.md
    scripts/rebuild_all.sh > logs/rebuild.log 2>&1
    touch logs/final_rebuild.done
    echo "- $(date '+%m-%d %H:%M') [auto] final rebuild done: $(cat logs/check_refs.log | tr '\n' ' ') $(head -1 logs/check_captions.log) consistency: $(head -3 logs/check_consistency.log | tr '\n' ' ')" >> ../STATUS.md
  fi
  sleep 7200
done
