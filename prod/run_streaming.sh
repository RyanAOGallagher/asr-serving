#!/usr/bin/env bash
# Qwen3-ASR STREAMING server (vLLM backend) -- ISOLATED from the production
# batched deployment in ~/ASR/Qwen3ASR (ports 8002/8003).
#
# DEFERRED LAUNCH: do NOT run this while the batched service holds the GPU.
# The 3090 Ti (24564 MiB) is ~95% used by the batched vLLM. This script
# refuses to start unless there is enough free VRAM, to protect production.

set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8005}"
HOST="${HOST:-0.0.0.0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.30}"     # fraction of TOTAL VRAM for vLLM
CHUNK_SIZE_SEC="${CHUNK_SIZE_SEC:-1.0}"  # lower = lower latency, more compute
UNFIXED_CHUNK_NUM="${UNFIXED_CHUNK_NUM:-4}"
UNFIXED_TOKEN_NUM="${UNFIXED_TOKEN_NUM:-5}"
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"     # preflight guard

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d "[:space:]")
echo "GPU free: ${free_mib} MiB (need >= ${MIN_FREE_MIB})"
if [ "${free_mib:-0}" -lt "$MIN_FREE_MIB" ]; then
  echo "ABORT: not enough free VRAM. The batched service is likely running."
  echo "Free the GPU first (stop batched, or lower its gpu_memory_utilization),"
  echo "or run this on another GPU host. Refusing to start to protect production."
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

exec ./.venv/bin/qwen-asr-demo-streaming \
  --asr-model-path "Qwen/Qwen3-ASR-1.7B" \
  --host "$HOST" --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --chunk-size-sec "$CHUNK_SIZE_SEC" \
  --unfixed-chunk-num "$UNFIXED_CHUNK_NUM" \
  --unfixed-token-num "$UNFIXED_TOKEN_NUM"
