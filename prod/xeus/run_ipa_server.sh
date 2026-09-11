#!/usr/bin/env bash
# Xeus / ZIPA / wav2vec2 IPA server on 3060-server-1 (192.168.1.239:8001).
# Mirrors the hand-launched process found running 2026-09-11 (started 2026-08-20):
#   cwd ~/ipa_asr, PYTHONPATH=~/ipa_asr/PhoneticXeus, XEUS_DTYPE=float16, venv uvicorn.
# Launch:  cd ~/ipa_asr && (setsid nohup ./run_ipa_server.sh >> ipa_server.log 2>&1 < /dev/null &)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-$PWD/PhoneticXeus}"
export XEUS_DTYPE="${XEUS_DTYPE:-float16}"
exec ./venv/bin/uvicorn ipa_server:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8001}"
