#!/usr/bin/env bash
# auto-token@2 on many layers under ONE registration (repo root, Git Bash):
#   bash scripts/auto2_all.sh "8,16,24,28,32,35,2,6,..."      (the scoring order; the passes read all of them at once)
# protocol-all ALONE -> hits-all (one pass) -> screen-all (one pass) -> prereg-all ALONE -> per layer, in the order given:
#   score-all on chunks of CHUNK layers (one model load each) -> per layer: export, summary commit, backup, artifacts
#   commit -> merge into main per chunk (MERGE_WT, optional)
# Resumable: registered manifests are not rewritten, finished passes are skipped, scored layers are skipped.
# GPU memory: honours BONSAI_GPU_FRACTION; hits/screen start at batch 2 under a cap; an out-of-memory error halves the
# batch and retries. A file data/auto2/STOP ends the run cleanly (the passes also check it between batches).
set -euo pipefail
ORDER="$1"
LAYERS="$(echo "$ORDER" | tr ',' '\n' | sort -n | paste -sd, -)"
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
push_branch() { for i in 1 2 3; do git push -q origin HEAD && return 0; sleep 30; done; echo "=== push failed (continuing)"; }
commit_alone() { git add "$1"; git commit -q -m "$2

$TRAILER"; push_branch; }

# run_pass NAME LOGNAME BATCH ARGS...: a GPU stage, retried at half the batch on an out-of-memory error
run_pass() {
  local name=$1 tag=$2 b=$3; shift 3
  local log="$LOGDIR/$tag.log" t0
  while :; do
    stop_check "$tag"
    t0=$(date +%s)
    if $CT "$name" --batch "$b" "$@" 2>&1 | tee "$log"; then
      echo "=== $tag done in $(( ($(date +%s) - t0) / 60 )) min (batch $b)"; return 0
    fi
    stop_check "$tag (after a failed or stopped run)"
    if grep -q "OutOfMemoryError" "$log" && [ "$b" -gt 1 ]; then
      b=$(( b / 2 )); echo "=== $tag ran out of memory: retrying at batch $b"
    else
      echo "=== $tag failed (see $log)"; exit 1
    fi
  done
}

merge_main() {  # in a separate worktree, so the running checkout's code never changes mid-run
  [ -n "${MERGE_WT:-}" ] || return 0
  for i in 1 2 3 4 5; do
    if (cd "$MERGE_WT" && git fetch -q origin && git checkout -q --detach origin/main \
        && git merge -q --no-ff "$BRANCH" -m "Merge auto-token@2 $1 ($BRANCH)" && git push -q origin HEAD:main); then
      echo "=== $1 merged into main"; return 0
    fi
    (cd "$MERGE_WT" && git merge --abort 2>/dev/null || true); sleep 60
  done
  echo "=== merging $1 into main failed 5 times (continuing; the next merge carries it)"
}

stop_check "registration"
echo "=== run of layers $LAYERS (scoring order $ORDER) at $(date -u +%H:%M:%S)"
if ! git log --format=%H -- graph/auto/bonsai1_multi.auto2.protocol.json | grep -q .; then
  $CT protocol-all --layers "$LAYERS"
  commit_alone graph/auto/bonsai1_multi.auto2.protocol.json "Pre-register auto-token@2 for layers $LAYERS (one manifest)"
fi
[ -e data/auto2/.hits-all.done ] || { run_pass hits-all hits-all "$HB" --layers "$LAYERS"; touch data/auto2/.hits-all.done; }
[ -e data/auto2/.screen-all.done ] || { run_pass screen-all screen-all "$HB" --layers "$LAYERS" ${SCREEN_ARGS:-}; touch data/auto2/.screen-all.done; }
if ! git log --format=%H -- graph/auto/bonsai1_multi.auto2.prereg.json | grep -q .; then
  stop_check "prereg-all"
  $CT prereg-all --layers "$LAYERS" | tee "$LOGDIR/prereg-all.log"
  commit_alone graph/auto/bonsai1_multi.auto2.prereg.json "Pre-register auto-token@2 tests for layers $LAYERS (every tests file's sha256)"
fi
# scoring in chunks (one model load each): regular checkpoints, and the map fills in chunk by chunk
CHUNK="${CHUNK:-5}"
todo=()
for L in $(echo "$ORDER" | tr ',' ' '); do
  git log --format=%H -- "graph/auto/bonsai1_L${L}.auto2.summary.json" | grep -q . && { echo "=== layer $L already committed: skipped"; continue; }
  todo+=("$L")
done
for ((i = 0; i < ${#todo[@]}; i += CHUNK)); do
  chunk=("${todo[@]:i:CHUNK}")
  csv="$(IFS=,; echo "${chunk[*]}")"
  run_pass score-all "score-$(IFS=-; echo "${chunk[*]}")" "$SB" --layers "$csv"
  for L in "${chunk[@]}"; do
    sum="graph/auto/bonsai1_L${L}.auto2.summary.json"
    $CT export-all --layers "$L" | tail -1
    held=$($PY -c "import json;s=json.load(open('$sum'));o=s['outcomes'];print(f\"{o.get('held',0)}/{sum(o.values())} tests held, {s['poles_held']} poles\")")
    commit_alone "$sum" "auto-token@2 layer $L scored: $held"
    $CT backup --layer "$L" | tail -1
    commit_alone "$sum" "auto-token@2 layer $L: artifacts backed up to releases (sha256 in the summary)"
    echo "=== layer $L done: $held | score $(grep -o '"minutes": [0-9.]*' "$sum" | head -1)"
  done
  merge_main "layers $csv"
  echo "=== chunk $csv checkpoint at $(date -u +%H:%M:%S)"
done
echo "=== all layers done"
