#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "usage: $0 CHECKPOINT OUTPUT_ROOT FIXED_WORKTREE [DECKS_JSON] [NAME_PREFIX]" >&2
  exit 2
fi

checkpoint=$(realpath "$1")
output_root="$2"
worktree=$(realpath "$3")
if [[ "$output_root" != /* ]] || [[ -e "$output_root" ]]; then
  echo "output root must be a new absolute path: $output_root" >&2
  exit 3
fi

python=/homes/lzhang/mypath/new/envs/trans/bin/python
sources=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/final_7d/sources.json
decks=${4:-/homes/lzhang/pocketmon/runs/experiment7-multideck-20260809/prepared/selection/selected_decks.json}
name_prefix=${5:-seed20260812_stabletie}
for required in "$checkpoint" "$sources" "$decks"; do
  if [[ ! -s "$required" ]]; then
    echo "required input is missing: $required" >&2
    exit 4
  fi
done

mkdir -p "$output_root"
portable="$output_root/universal_bc.npz"
parity="$output_root/portable_parity.json"
packages="$output_root/packages"
cd "$worktree"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$python" experiment7/integration/export_and_package.py export \
  --checkpoint "$checkpoint" \
  --output "$portable"
"$python" experiment7/integration/export_and_package.py verify-universal \
  --reference-root "$worktree/experiment7/reference" \
  --sources "$sources" \
  --checkpoint "$checkpoint" \
  --portable "$portable" \
  --output "$parity" \
  --python "$python" \
  --decisions-per-source 150
"$python" experiment7/integration/export_and_package.py package-universal \
  --reference-root "$worktree/experiment7/reference" \
  --sources "$sources" \
  --decks "$decks" \
  --portable "$portable" \
  --output-root "$packages" \
  --name-prefix "$name_prefix"
sha256sum "$checkpoint" "$portable" "$parity" "$packages/packages.json" >"$output_root/SHA256SUMS"
touch "$output_root/SUCCESS"
echo "BC_PACKAGE_SUCCESS output=$output_root"
