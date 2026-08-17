#!/usr/bin/env bash
set -Eeuo pipefail

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
launcher=/homes/lzhang/worktrees/experiment7-a08-learner-clean-8885a59
python=/homes/lzhang/mypath/new/envs/trans/bin/python
tool="$launcher/experiment7/integration/export_and_package.py"

package_one() {
  chain="$1"
  local_checkpoint="$2"
  generation_root="$root/learners/$chain/generation-000000-bootstrap"
  deployment="$generation_root/deployment"
  local_portable="/dev/shm/$chain-bootstrap-20260814.npz"
  mkdir -p "$deployment/packages"
  env PYTHONNOUSERSITE=1 "$python" -s "$tool" export \
    --checkpoint "$local_checkpoint" --output "$local_portable"
  temporary="$deployment/.universal_ppo.npz.tmp-$$-$RANDOM"
  cp "$local_portable" "$temporary"
  mv "$temporary" "$deployment/universal_ppo.npz"
  env PYTHONNOUSERSITE=1 "$python" -s "$tool" package-universal \
    --reference-root /homes/lzhang/pocketmon-worktrees/experiment7-248c61b/experiment7/reference \
    --sources /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/final_7d/sources.json \
    --decks "$deployment/deck.json" \
    --portable "$deployment/universal_ppo.npz" \
    --output-root "$deployment/packages" \
    --name-prefix "live_${chain}_g000000"
  echo "BOOTSTRAP_PACKAGED chain=$chain manifest=$deployment/packages/packages.json"
}

case "${1:?chain required}" in
  universal_ppo_standard_1m)
    package_one "$1" /tmp/experiment7-bc-d14-local-20260813/standard_1m/best_model.pt
    ;;
  universal_ppo_large_256x6|lucario_gold_exact)
    package_one "$1" /tmp/experiment7-bc-d14-local-20260813/large_256x6/best_model.pt
    ;;
  *) echo "unsupported chain: $1" >&2; exit 2 ;;
esac
