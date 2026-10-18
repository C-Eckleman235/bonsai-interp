#!/usr/bin/env bash
# Download the tokenized corpus and the full corpus scan from the GitHub release into ./data
set -e
cd "$(dirname "$0")/.."
mkdir -p data
gh release download data-v1 --repo moridinamael/bonsai-interp --dir data --pattern '*.npz' --clobber
ls -la data
