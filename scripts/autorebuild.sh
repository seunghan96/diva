#!/bin/bash
# Re-run rebuild_all every 2 h (if none running) so finished experiments flow into tables/figures automatically.
cd .
while true; do
  sleep 7200
  if ! ps -eo args | grep -v grep | grep -q "scripts/rebuild_all.sh"; then
    scripts/rebuild_all.sh > logs/rebuild.log 2>&1
    echo "- $(date '+%Y-%m-%d %H:%M') — autorebuild done: $(cat logs/check_refs.log | tr '\n' ' ') $(cat logs/check_captions.log | head -1)" >> ../STATUS.md
  fi
done
