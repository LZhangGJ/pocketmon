#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 ROLE GENERATION PPO_ROOT OUTPUT_ROOT [FIXED_WORKTREE]" >&2
  exit 2
fi

role="$1"
generation="$2"
ppo_root=$(realpath "$3")
output_root="$4"
worktree="${5:-/homes/lzhang/worktrees/experiment7-13c2fe33d9e1}"
case "$role" in
  generalist|hard_exploiter|diversity|conservative) ;;
  *) echo "unsupported role: $role" >&2; exit 2 ;;
esac
if [[ ! "$generation" =~ ^[0-9]+$ ]] || (( generation < 1 || generation > 40 )); then
  echo "generation must be in [1, 40]: $generation" >&2
  exit 2
fi
if [[ "$output_root" != /* ]] || [[ -e "$output_root" ]]; then
  echo "output root must be a new absolute path: $output_root" >&2
  exit 3
fi

worktree=$(realpath "$worktree")
python=/homes/lzhang/mypath/new/envs/trans/bin/python
sources=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/final_7d/sources.json
decks=/homes/lzhang/pocketmon/runs/experiment7-multideck-20260809/prepared/selection/selected_decks.json
checkpoint=$(printf "%s/%s/generation-%04d/checkpoint.pt" "$ppo_root" "$role" "$generation")
metrics=$(printf "%s/%s/generation-%04d/metrics.json" "$ppo_root" "$role" "$generation")
rollouts=$(printf "%s/%s/generation-%04d/rollouts.jsonl.gz" "$ppo_root" "$role" "$generation")
for required in "$checkpoint" "$metrics" "$rollouts" "$sources" "$decks"; do
  if [[ ! -s "$required" ]]; then
    echo "required completed input is missing: $required" >&2
    exit 4
  fi
done

mkdir -p "$output_root"
portable="$output_root/universal_ppo.npz"
parity="$output_root/portable_parity.json"
packages="$output_root/packages"
name_prefix=$(printf "%s_g%04d" "$role" "$generation")
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
echo "PACKAGE_SUCCESS role=$role generation=$generation output=$output_root"
