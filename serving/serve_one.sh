#!/bin/bash
# usage: serve_one.sh MODEL SHORT PORT UTIL MAXLEN MAXSEQS TP QUANT [MNBT]   (CUDA_VISIBLE_DEVICES set by fleet.py's locked launcher)
# Memory budget (UTIL = alloc/total) is decided inside the per-GPU flock by fleet.py (free read twice, margin 6/10 GB).
# Prefill-bound workload (DESIGN.md s7): --max-num-batched-tokens "$MNBT" + prefix caching.
# Env follows forecastbrief/scripts/vllm_serve_m3.sh. Served under both the HF id and the short name.
MODEL=$1; SHORT=$2; PORT=$3; UTIL=$4; MAXLEN=$5; MAXSEQS=$6; TP=$7; QUANT=$8; MNBT=${9:-16384}
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export PATH=<path>:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_HOME=<path>
export PATH=$CUDA_HOME/bin:$PATH
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK" FLASHINFER_EXTRA_CFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export FLASHINFER_EXTRA_LDFLAGS="-L<path>" FLASHINFER_CUDA_ARCH_LIST=8.9 MAX_JOBS=4
Q=(); [ -n "$QUANT" ] && [ "$QUANT" != "none" ] && Q=(--quantization "$QUANT")
echo "host=$(hostname -s) gpus=$CUDA_VISIBLE_DEVICES model=$MODEL port=$PORT util=$UTIL maxlen=$MAXLEN tp=$TP quant=${QUANT:-none} start=$(date +%s)"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" "$SHORT" --host 0.0.0.0 --port "$PORT" \
  --gpu-memory-utilization "$UTIL" --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" \
  --tensor-parallel-size "$TP" --max-logprobs 20 --max-num-batched-tokens "$MNBT" --enable-prefix-caching "${Q[@]}"
