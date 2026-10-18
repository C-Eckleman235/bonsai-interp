#!/usr/bin/env bash
# Download model weights into ./models.
#   bash scripts/download_models.sh          Bonsai-8B only (about 16 GB): all you need to play
#   bash scripts/download_models.sh --all    the whole precision ladder: + Ternary-Bonsai-8B and Qwen3-8B (about 48 GB)
set -e
cd "$(dirname "$0")/.."
repos=(prism-ml/Bonsai-8B-unpacked)
[ "$1" = "--all" ] && repos+=(prism-ml/Ternary-Bonsai-8B-unpacked Qwen/Qwen3-8B)
for r in "${repos[@]}"; do
  uvx --from "huggingface_hub[cli]" hf download "$r" --local-dir "models/${r#*/}"
done
