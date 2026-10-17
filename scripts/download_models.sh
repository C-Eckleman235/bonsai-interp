#!/usr/bin/env bash
# Download the 8B precision ladder into ./models
set -e
cd "$(dirname "$0")/.."
for r in prism-ml/Bonsai-8B-unpacked prism-ml/Ternary-Bonsai-8B-unpacked Qwen/Qwen3-8B; do
  uvx --from "huggingface_hub[cli]" hf download "$r" --local-dir "models/${r#*/}"
done
