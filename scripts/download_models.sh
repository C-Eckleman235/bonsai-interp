#!/usr/bin/env bash
# Download model weights into ./models.
#   bash scripts/download_models.sh             Bonsai-8B as PrismML's 1-bit GGUF (about 1.2 GB): all you need to play
#   bash scripts/download_models.sh --unpacked  Bonsai-8B as unpacked safetensors (about 16 GB; same weights, bit for bit)
#   bash scripts/download_models.sh --all       the whole precision ladder: + Ternary-Bonsai-8B and Qwen3-8B (about 48 GB)
set -e
cd "$(dirname "$0")/.."
hf() { uvx --from "huggingface_hub[cli]" hf "$@"; }
if [ "$1" = "--unpacked" ] || [ "$1" = "--all" ]; then
  repos=(prism-ml/Bonsai-8B-unpacked)
  [ "$1" = "--all" ] && repos+=(prism-ml/Ternary-Bonsai-8B-unpacked Qwen/Qwen3-8B)
  for r in "${repos[@]}"; do hf download "$r" --local-dir "models/${r#*/}"; done
  exit 0
fi
# the GGUF holds every weight; config and tokenizer come from the unpacked repository (a few MB)
hf download prism-ml/Bonsai-8B-gguf Bonsai-8B-Q1_0.gguf --local-dir models/Bonsai-8B
hf download prism-ml/Bonsai-8B-unpacked --local-dir models/Bonsai-8B \
  --include "config.json" "generation_config.json" "tokenizer.json" "tokenizer_config.json" "vocab.json" "merges.txt" \
            "added_tokens.json" "special_tokens_map.json" "chat_template.jinja" "LICENSE" "NOTICE.txt"
