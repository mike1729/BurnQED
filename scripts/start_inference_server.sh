#!/bin/bash
# Launch BurnQED inference server (delegates to raw SGLang).
#
# Previously used python/inference_server.py with sgl.Engine, but
# SGLang Issue #8066 breaks batch return_hidden_states. Now uses raw
# SGLang server + separate encode server for embeddings.
#
# Usage:
#   ./scripts/start_inference_server.sh [model_path]
#   PORT=30000 TP=2 ./scripts/start_inference_server.sh
#
# Prerequisites:
#   pip install "sglang[all]"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "${REPO_ROOT}/scripts/start_sglang.sh" "$@"
