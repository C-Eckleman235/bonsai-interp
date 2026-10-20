#!/usr/bin/env bash
# auto-token@2 on each layer given, one at a time, with the pre-registration commits in order:
#   protocol manifest ALONE -> hits -> screen -> prereg manifest ALONE -> score -> export -> summary -> backup -> summary.
# usage (repo root, Git Bash): bash scripts/auto2_layers.sh 4 8 16      stops at the first error; never layer 12
# Each layer takes about 75 GPU minutes on an RTX 5080 (FINDINGS 2026-10-20). Under a BONSAI_GPU_FRACTION cap,
# hits and screen need batch 2 (set HITS_BATCH / SCORE_BATCH to override).
set -euo pipefail
PY="${PY:-uv run python}"
CT="$PY -m bonsai_interp.auto.context_tier"
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HB="${HITS_BATCH:-$(grep -q '^BONSAI_GPU_FRACTION=' .env 2>/dev/null && echo 2 || echo 8)}"
SB="${SCORE_BATCH:-16}"
TRAILER="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
commit_alone() { git add "$1"; git commit -q -m "$2

$TRAILER"; git push -q origin HEAD; }
for L in "$@"; do
  [ "$L" = "12" ] && { echo "refusing layer 12 (it was used to look)"; exit 1; }
  echo "=== layer $L start $(date -u +%H:%M:%S)"
  man="graph/auto/bonsai1_L${L}.auto2.protocol.json"
  if ! git log --format=%H -- "$man" | grep -q .; then
    $CT protocol --layer "$L"
    commit_alone "$man" "Pre-register auto-token@2, layer $L"
  fi
  t0=$(date +%s); $CT hits --layer "$L" --batch "$HB"; t1=$(date +%s)
  $CT screen --layer "$L" --batch "$HB"; t2=$(date +%s)
  $CT prereg --layer "$L"; t3=$(date +%s)
  commit_alone "graph/auto/bonsai1_L${L}.auto2.prereg.json" "Pre-register auto-token@2 tests, layer $L (tests sha256 in the manifest)"
  $CT score --layer "$L" --batch "$SB"; t4=$(date +%s)
  $CT export --layer "$L"
  sum="graph/auto/bonsai1_L${L}.auto2.summary.json"
  held=$($PY -c "import json;s=json.load(open('$sum'));o=s['outcomes'];print(f\"{o.get('held',0)}/{sum(o.values())} tests held, {s['poles_held']} poles\")")
  commit_alone "$sum" "auto-token@2 layer $L scored: $held"
  $CT backup --layer "$L"
  commit_alone "$sum" "auto-token@2 layer $L: artifacts backed up to releases (sha256 in the summary)"
  echo "=== layer $L done: $held | minutes: hits $(( (t1-t0)/60 )), screen $(( (t2-t1)/60 )), prereg $(( (t3-t2)/60 )), score $(( (t4-t3)/60 ))"
done
