#!/bin/bash
# Prints allowed GPUs that look idle (util<5% and free>20GB) on two consecutive checks, 180 s apart.
# Uses the shared gpu_check.sh "pit" flag so that <other>/<other> GPUs are never reported as fillable.
HOSTS="02-l40 03-l40 05-l40s 06-l40s 07-l40s 09-l40s 11-l40s 12-l40s 14-l40s 15-l40s 16-l40s 27-l40s 31-l40s 32-l40s 33-l40s 34-l40s"
declare -A prev
while true; do
  cur=""
  for h in $HOSTS; do
    out=$(timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=8 <host>$h.onprem \
      "nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits; bash <path> 2>/dev/null | awk '\$1==\"GPU\"{print \"PIT\",\$2,(\$4>0||\$5>0)?1:0}'" 2>/dev/null)
    while IFS=', ' read -r a b c; do
      [ "$a" = "PIT" ] && continue
      [ -z "$c" ] && continue
      pit=$(echo "$out" | awk -v g=$a '$1=="PIT" && $2==g {print $3}')
      if [ "${c%.*}" -lt 5 ] && [ "${b%.*}" -gt 20000 ] && [ "${pit:-0}" = "0" ]; then cur="$cur <host>$h:g$a(free=${b})"; fi
    done <<< "$(echo "$out" | grep -v '^PIT')"
  done
  both=""
  for x in $cur; do k=${x%%(*}; [ -n "${prev[$k]}" ] && both="$both $x"; done
  unset prev; declare -A prev
  for x in $cur; do prev[${x%%(*}]=1; done
  [ -n "$both" ] && echo "$(date +%H:%M) IDLE:$both"
  sleep 180
done
