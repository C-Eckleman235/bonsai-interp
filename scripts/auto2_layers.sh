#!/usr/bin/env bash
# auto-token@2 on each layer given, one at a time, with the pre-registration commits in order:
#   protocol manifest ALONE -> hits -> screen -> prereg manifest ALONE -> score -> export -> summary -> backup -> summary
#   [-> merge into main with a merge commit, when MERGE_WT names a separate worktree to merge in]
# usage (repo root, Git Bash): bash scripts/auto2_layers.sh 4 8 16      stops at the first error; never layer 12
# Each layer takes about 75 GPU minutes on an RTX 5080 (FINDINGS 2026-10-20). Under a BONSAI_GPU_FRACTION cap, hits
# and screen start at batch 2; any stage that runs out of memory is retried at half the batch (never a higher cap).
# A file data/auto2/STOP ends the run cleanly before the next stage.
set -euo pipefail
PY="${PY:-uv run python}"
CT="$PY -m bonsai_interp.auto.context_tier"
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HB="${HITS_BATCH:-$(grep -q '^BONSAI_GPU_FRACTION=' .env 2>/dev/null && echo 2 || echo 8)}"
SB="${SCORE_BATCH:-16}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
TRAILER="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
LOGDIR="${LOGDIR:-data/auto2/logs}"
mkdir -p "$LOGDIR"

stop_check() { if [ -e data/auto2/STOP ]; then echo "=== STOP file found before $1: stopping cleanly"; exit 0; fi; }
commit_alone() { git add "$1"; git commit -q -m "$2

$TRAILER"; push_branch; }
push_branch() { for i in 1 2 3; do git push -q origin HEAD && return 0; sleep 30; done; echo "push failed (continuing)"; }

# run_stage NAME LAYER BATCH: a GPU stage, retried at half the batch on an out-of-memory error
run_stage() {
  local name=$1 L=$2 b=$3 log
  while :; do
    stop_check "$name L$L"
    log="$LOGDIR/L${L}.${name}.log"
    if $CT "$name" --layer "$L" --batch "$b" 2>&1 | tee "$log"; then  # pipefail: the stage's own exit status
      return 0
    fi
    if grep -q "OutOfMemoryError" "$log" && [ "$b" -gt 1 ]; then
      b=$(( b / 2 )); echo "=== $name L$L ran out of memory: retrying at batch $b"
    else
      echo "=== $name L$L failed (see $log)"; return 1
    fi
  done
}

merge_main() {  # in a separate worktree, so the running checkout's code never changes mid-run
  [ -n "${MERGE_WT:-}" ] || return 0
  for i in 1 2 3 4 5; do
    if (cd "$MERGE_WT" && git fetch -q origin && git checkout -q --detach origin/main \
        && git merge -q --no-ff "$BRANCH" -m "Merge auto-token@2 layer $1 ($BRANCH)" && git push -q origin HEAD:main); then
      echo "=== layer $1 merged into main"; return 0
    fi
    (cd "$MERGE_WT" && git merge --abort 2>/dev/null || true); sleep 60
  done
  echo "=== merging layer $1 into main failed 5 times (continuing; the next layer's merge carries it)"
}

for L in "$@"; do
  [ "$L" = "12" ] && { echo "refusing layer 12 (it was used to look)"; exit 1; }
  stop_check "layer $L"
  echo "=== layer $L start $(date -u +%H:%M:%S)"
  man="graph/auto/bonsai1_L${L}.auto2.protocol.json"
  if ! git log --format=%H -- "$man" | grep -q .; then
    $CT protocol --layer "$L"
    commit_alone "$man" "Pre-register auto-token@2, layer $L"
  fi
  t0=$(date +%s); run_stage hits "$L" "$HB"; t1=$(date +%s)
  run_stage screen "$L" "$HB"; t2=$(date +%s)
  stop_check "prereg L$L"
  $CT prereg --layer "$L" | tee "$LOGDIR/L${L}.prereg.log"; t3=$(date +%s)
  commit_alone "graph/auto/bonsai1_L${L}.auto2.prereg.json" "Pre-register auto-token@2 tests, layer $L (tests sha256 in the manifest)"
  run_stage score "$L" "$SB"; t4=$(date +%s)
  $CT export --layer "$L"
  sum="graph/auto/bonsai1_L${L}.auto2.summary.json"
  held=$($PY -c "import json;s=json.load(open('$sum'));o=s['outcomes'];print(f\"{o.get('held',0)}/{sum(o.values())} tests held, {s['poles_held']} poles\")")
  commit_alone "$sum" "auto-token@2 layer $L scored: $held"
  $CT backup --layer "$L"
  commit_alone "$sum" "auto-token@2 layer $L: artifacts backed up to releases (sha256 in the summary)"
  merge_main "$L"
  echo "=== layer $L done: $held | minutes: hits $(( (t1-t0)/60 )), screen $(( (t2-t1)/60 )), prereg $(( (t3-t2)/60 )), score $(( (t4-t3)/60 ))"
done
echo "=== all layers done"
