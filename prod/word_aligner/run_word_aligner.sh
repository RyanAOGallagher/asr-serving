#!/usr/bin/env bash
# Word-level Viterbi forced aligner (XLS-R char CTC), brought from research(runpod).
# Streaming WS server on /stream. ISOLATED + DEFERRED: refuses to start unless GPU is free.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8009}"
HOST="${HOST:-0.0.0.0}"
export ALIGN_CKPT="${ALIGN_CKPT:-$(pwd)/align_xlsr_best}"
export ALIGN_DEVICE="${ALIGN_DEVICE:-cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-2500}"   # XLS-R ~1.5GB on GPU

if [ "$ALIGN_DEVICE" = "cuda" ]; then
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d "[:space:]")
  echo "GPU free: ${free_mib} MiB (need >= ${MIN_FREE_MIB})"
  if [ "${free_mib:-0}" -lt "$MIN_FREE_MIB" ]; then
    echo "ABORT: not enough free VRAM (batched service likely running)."
    echo "Free the GPU, run another host, or set ALIGN_DEVICE=cpu. Refusing to protect production."
    exit 1
  fi
fi

echo "ckpt=$ALIGN_CKPT device=$ALIGN_DEVICE port=$PORT"
exec ../.venv/bin/uvicorn word_align_server:app --host "$HOST" --port "$PORT"
