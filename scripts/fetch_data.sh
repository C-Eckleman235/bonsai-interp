#!/usr/bin/env bash
# Download data from the GitHub releases into ./data
#   bash scripts/fetch_data.sh          the tokenized corpus and the full corpus scan (data-v1, about 1.4 GB)
#   bash scripts/fetch_data.sh atlas    the atlas: one table per layer of every pole's corpus word hits and auto screen
#                                       (Arrow IPC, zstd; about 70 MB) -> data/atlas/, read by `play card` and the map
#   bash scripts/fetch_data.sh all      both
set -e
cd "$(dirname "$0")/.."
mkdir -p data
what="${1:-corpus}"
if [ "$what" = corpus ] || [ "$what" = all ]; then
  gh release download data-v1 --repo moridinamael/bonsai-interp --dir data --pattern '*.npz' --clobber
fi
if [ "$what" = atlas ] || [ "$what" = all ]; then
  mkdir -p data/atlas
  # the releases are public: plain curl works without logging in to gh
  if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    gh release download atlas --repo moridinamael/bonsai-interp --dir data/atlas --pattern '*.arrow' --clobber
  else
    for f in $(curl -sSL https://api.github.com/repos/moridinamael/bonsai-interp/releases/tags/atlas | grep -o '"browser_download_url": *"[^"]*\.arrow"' | sed 's/.*"\(https[^"]*\)"/\1/'); do
      curl -sSL -o "data/atlas/$(basename "$f")" "$f"
    done
  fi
  ls data/atlas | wc -l | xargs echo "atlas files:"
fi
ls -la data
