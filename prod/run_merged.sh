#!/usr/bin/env bash
# Merged Qwen3-ASR: /transcribe (file, short/long multipath, optional timestamps) + /api streaming.
# ONE model instance. ISOLATED + DEFERRED: refuses to start without GPU headroom.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8006}"; HOST="${HOST:-0.0.0.0}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"
export TORN_P="${TORN_P:-0.93}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export ST_CHUNK_SIZE_SEC="${ST_CHUNK_SIZE_SEC:-1.0}"
export ST_UNFIXED_CHUNK_NUM="${ST_UNFIXED_CHUNK_NUM:-9999}"
export LONG_LOCK_GROUP="${LONG_LOCK_GROUP:-1}"   # long jobs yield after every chunk so interactive short clips run first
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"   # full Qwen+aligner footprint
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d "[:space:]")
echo "GPU free: ${free_mib} MiB (need >= ${MIN_FREE_MIB})"
if [ "${free_mib:-0}" -lt "$MIN_FREE_MIB" ]; then
  echo "ABORT: not enough free VRAM (batched service likely running). Refusing to protect production."
  exit 1
fi
exec ./.venv/bin/uvicorn server_merged:app --host "$HOST" --port "$PORT"
