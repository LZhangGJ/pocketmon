#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/homes/lzhang/mypath/new/envs/trans/bin/python}"
RUN_ID="${RUN_ID:-rl-bc-004-tx8-$(hostname)}"
RUN_ROOT="${RUN_ROOT:-/homes/lzhang/pocketmon/runs/${RUN_ID}}"

SOURCE_REPLAY="${SOURCE_REPLAY:-/homes/lzhang/pocketmon/data/processed/public_replay_2026-08-05.jsonl.gz}"
SOURCE_DECK_MAP="${SOURCE_DECK_MAP:-/homes/lzhang/pocketmon/data/processed/replay_decks_2026-08-05.jsonl.gz}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/homes/lzhang/rl_bc_003_outputs/seed_17/checkpoints/seed_17_best.pt}"
CARD_DATABASE="${CARD_DATABASE:-${ROOT}/data/reference/official_cards.json}"
ATTACK_DATABASE="${ATTACK_DATABASE:-${ROOT}/data/reference/official_attacks.json}"
REFERENCE_NOTEBOOK="${REFERENCE_NOTEBOOK:-${ROOT}/research/kaggle_intelligence/2026-08-08/notebooks/source/07__tetsutani__grimmsnarl-ex-damage-transfer-control.ipynb}"
REFERENCE_DECK="${REFERENCE_DECK:-}"
GENERIC_FRACTION="${GENERIC_FRACTION:-0.15}"
SEED="${SEED:-17}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_FORMAL="${RUN_FORMAL:-1}"
CG_DIR="${CG_DIR:-}"
OPPONENT_MANIFEST="${OPPONENT_MANIFEST:-${ROOT}/configs/opponent_pool_frontier_server.json}"
GATE_GAMES_PER_OPPONENT="${GATE_GAMES_PER_OPPONENT:-2}"

for path in \
    "${PYTHON}" \
    "${SOURCE_REPLAY}" \
    "${SOURCE_DECK_MAP}" \
    "${BASE_CHECKPOINT}" \
    "${CARD_DATABASE}" \
    "${ATTACK_DATABASE}"; do
    if [[ ! -e "${path}" ]]; then
        echo "required path is missing: ${path}" >&2
        exit 2
    fi
done

if [[ -e "${RUN_ROOT}" ]]; then
    echo "refusing to overwrite run root: ${RUN_ROOT}" >&2
    exit 2
fi
mkdir -p \
    "${RUN_ROOT}/reference" \
    "${RUN_ROOT}/data" \
    "${RUN_ROOT}/config" \
    "${RUN_ROOT}/smoke/checkpoints" \
    "${RUN_ROOT}/formal/checkpoints" \
    "${RUN_ROOT}/packages" \
    "${RUN_ROOT}/gate"

cd "${ROOT}"

echo "[1/8] static checks"
"${PYTHON}" -m py_compile \
    rl/temporal_model.py \
    rl/temporal_agent_adapter.py \
    scripts/select_modal_deck.py \
    scripts/write_temporal_specialist_config.py \
    scripts/train_rl_bc_004_transformer8.py \
    scripts/materialize_temporal_specialist_agent.py
"${PYTHON}" -m pytest -q tests/test_temporal_transformer.py

if [[ -z "${REFERENCE_DECK}" ]]; then
    echo "[2/8] materialize recent Grimmsnarl reference deck"
    "${PYTHON}" scripts/materialize_notebook_agent.py \
        "${REFERENCE_NOTEBOOK}" \
        "${RUN_ROOT}/reference/grimmsnarl"
    REFERENCE_DECK="${RUN_ROOT}/reference/grimmsnarl/deck.csv"
else
    echo "[2/8] use supplied reference deck: ${REFERENCE_DECK}"
fi

echo "[3/8] select modal exact list within the Grimmsnarl Pokemon core"
"${PYTHON}" scripts/select_modal_deck.py \
    --deck-map "${SOURCE_DECK_MAP}" \
    --reference-deck "${REFERENCE_DECK}" \
    --card-database "${CARD_DATABASE}" \
    --output "${RUN_ROOT}/data/target_deck.csv" \
    --audit-output "${RUN_ROOT}/data/modal_deck_audit.json"

echo "[4/8] build fixed-deck specialist replay dataset"
"${PYTHON}" scripts/build_deck_specialist_dataset.py \
    --input "${SOURCE_REPLAY}" \
    --deck-map "${SOURCE_DECK_MAP}" \
    --target-deck "${RUN_ROOT}/data/target_deck.csv" \
    --match exact \
    --generic-fraction "${GENERIC_FRACTION}" \
    --seed 20260808 \
    --card-database "${CARD_DATABASE}" \
    --output "${RUN_ROOT}/data/replay.jsonl.gz" \
    --output-deck-map "${RUN_ROOT}/data/decks.jsonl.gz" \
    --audit-output "${RUN_ROOT}/data/dataset_audit.json"

EXPERIMENT_ID="RL-BC-004-TX8-${RUN_ID}"
echo "[5/8] freeze planned config"
"${PYTHON}" scripts/write_temporal_specialist_config.py \
    --experiment-id "${EXPERIMENT_ID}" \
    --input "${RUN_ROOT}/data/replay.jsonl.gz" \
    --deck-map "${RUN_ROOT}/data/decks.jsonl.gz" \
    --target-deck "${RUN_ROOT}/data/target_deck.csv" \
    --initialize-from "${BASE_CHECKPOINT}" \
    --card-database "${CARD_DATABASE}" \
    --attack-database "${ATTACK_DATABASE}" \
    --seed "${SEED}" \
    --epochs 12 \
    --history-length 8 \
    --output "${RUN_ROOT}/config/planned.json"

COMMON_ARGS=(
    --input "${RUN_ROOT}/data/replay.jsonl.gz"
    --deck-map "${RUN_ROOT}/data/decks.jsonl.gz"
    --target-deck "${RUN_ROOT}/data/target_deck.csv"
    --planned-config "${RUN_ROOT}/config/planned.json"
    --initialize-from "${BASE_CHECKPOINT}"
    --experiment-id "${EXPERIMENT_ID}"
    --seed "${SEED}"
    --epochs 12
    --history-length 8
    --card-database "${CARD_DATABASE}"
    --attack-database "${ATTACK_DATABASE}"
    --device "cuda:0"
)

echo "[6/8] two-batch smoke"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" \
    scripts/train_rl_bc_004_transformer8.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint-dir "${RUN_ROOT}/smoke/checkpoints" \
    --metrics-output "${RUN_ROOT}/smoke/metrics.json" \
    --split-output "${RUN_ROOT}/smoke/split.json" \
    --max-batches 2

if [[ "${RUN_FORMAL}" != "1" ]]; then
    echo "smoke completed; RUN_FORMAL=${RUN_FORMAL}, stopping before formal training"
    exit 0
fi

echo "[7/8] formal 12-epoch training"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" \
    scripts/train_rl_bc_004_transformer8.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint-dir "${RUN_ROOT}/formal/checkpoints" \
    --metrics-output "${RUN_ROOT}/formal/metrics.json" \
    --split-output "${RUN_ROOT}/formal/split.json"

BEST_CHECKPOINT="$(
    "${PYTHON}" - "${RUN_ROOT}/formal/metrics.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["checkpoint"]["path"])
PY
)"

"${PYTHON}" scripts/materialize_temporal_specialist_agent.py \
    --checkpoint "${BEST_CHECKPOINT}" \
    --deck "${RUN_ROOT}/data/target_deck.csv" \
    --output "${RUN_ROOT}/packages/candidate" \
    --name "${EXPERIMENT_ID}"

echo "[8/8] local diagnostic gate"
if [[ -n "${CG_DIR}" && -d "${CG_DIR}" && -f "${OPPONENT_MANIFEST}" ]]; then
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${PYTHON}" \
        scripts/run_opponent_pool.py \
        --target "${RUN_ROOT}/packages/candidate" \
        --cg-dir "${CG_DIR}" \
        --manifest "${OPPONENT_MANIFEST}" \
        --games "${GATE_GAMES_PER_OPPONENT}" \
        --output "${RUN_ROOT}/gate/opponent_pool.json"
else
    cat > "${RUN_ROOT}/gate/NOT_RUN.txt" <<EOF
Training and packaging artifacts are complete.
Run the gate directly after setting CG_DIR:
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ${PYTHON} \
    scripts/run_opponent_pool.py \
    --target ${RUN_ROOT}/packages/candidate \
    --cg-dir /path/to/cg \
    --manifest ${OPPONENT_MANIFEST} \
    --games ${GATE_GAMES_PER_OPPONENT} \
    --output ${RUN_ROOT}/gate/opponent_pool.json
EOF
fi

cat <<EOF
RL-BC-004 run completed
  run root:       ${RUN_ROOT}
  target deck:    ${RUN_ROOT}/data/target_deck.csv
  planned config: ${RUN_ROOT}/config/planned.json
  metrics:        ${RUN_ROOT}/formal/metrics.json
  candidate:      ${RUN_ROOT}/packages/candidate
  local gate:     ${RUN_ROOT}/gate
EOF
