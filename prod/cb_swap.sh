#!/usr/bin/env bash
# Swap the merged server for stock qwen-asr-serve (vLLM continuous batching) and back.
# Usage: ./cb_swap.sh start   # stops server_merged, starts qwen-asr-serve on :8021
#        ./cb_swap.sh stop    # kills qwen-asr-serve, restarts server_merged
# NOTE: 'start' takes ports 8006 consumers (anchor/smart/streaming) DOWN until 'stop'.
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-}" in
  start)
    pkill -f 'uvicorn server_merged:app' || true
    sleep 3; pkill -9 -f 'VLLM::EngineCore' || true
    sleep 5
    nohup ./.venv/bin/qwen-asr-serve Qwen/Qwen3-ASR-1.7B \
      --gpu-memory-utilization 0.55 --max-model-len 4096 \
      --host 0.0.0.0 --port 8021 > logs/cb_serve.log 2>&1 &
    echo "qwen-asr-serve starting on :8021 (pid $!). tail -f logs/cb_serve.log; ready when 'Application startup complete'."
    ;;
  stop)
    pkill -f 'qwen-asr-serve|vllm serve.*8021' || true
    sleep 3; pkill -9 -f 'VLLM::EngineCore' || true
    sleep 5
    nohup ./run_merged.sh > logs/merged_restart.log 2>&1 &
    echo 'server_merged restarting on :8006. tail -f logs/merged_restart.log'
    ;;
  *) echo 'usage: cb_swap.sh start|stop'; exit 1;;
esac
