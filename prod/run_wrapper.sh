#!/usr/bin/env bash
# New isolated wrapper on 8002: /transcribe proxy (+ElevenLabs fallback +Supabase/CDN log)
# and WS /stream proxy. Backend = merged server (default :8006). Prod stt_wrapper.py untouched.
set -euo pipefail
cd "$(dirname "$0")"
export ASR_BACKEND_URL="${ASR_BACKEND_URL:-http://127.0.0.1:8006}"
export WRAPPER_ENV_FILE="${WRAPPER_ENV_FILE:-/home/ailab/ASR/Qwen3ASR/.env}"
exec ./.venv/bin/uvicorn wrapper_merged:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8002}"
