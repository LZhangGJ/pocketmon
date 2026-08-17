#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/homes/lzhang/pocketmon}
PYTHON=${PYTHON:-/homes/lzhang/mypath/new/envs/trans/bin/python}
COMMIT=${1:?usage: linux_bootstrap.sh COMMIT [WORKTREE]}
WORKTREE=${2:-/homes/lzhang/worktrees/experiment7-${COMMIT:0:12}}

cd "$REPO"
git fetch origin --prune
git cat-file -e "${COMMIT}^{commit}"
if [[ ! -d "$WORKTREE/.git" && ! -f "$WORKTREE/.git" ]]; then
  mkdir -p "$(dirname "$WORKTREE")"
  git worktree add --detach "$WORKTREE" "$COMMIT"
fi
cd "$WORKTREE"
actual=$(git rev-parse HEAD)
[[ "$actual" == "$COMMIT" ]] || { echo "worktree commit mismatch: $actual != $COMMIT" >&2; exit 2; }

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" -m compileall -q experiment7/reference experiment7/integration tests
"$PYTHON" -m unittest discover -s tests -p 'test_*.py'

printf '{"host":"%s","commit":"%s","worktree":"%s","python":"%s"}\n' \
  "$(hostname)" "$COMMIT" "$WORKTREE" "$PYTHON"
