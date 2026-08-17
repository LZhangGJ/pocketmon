#!/usr/bin/env bash
set -u
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
echo MATRIX_FILE
ls -l --time-style=long-iso "${root}/monitoring/full-matrix/latest.json" 2>/dev/null || true
python3 - <<'PY'
import json
from pathlib import Path
p = Path('/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/monitoring/full-matrix/latest.json')
if p.exists():
    d = json.loads(p.read_text())
    print('MATRIX_META', {k: d.get(k) for k in ('createdAt','completedAt','status','round','agentCount','games')})
    print('MATRIX_KEYS', sorted(d)[:40])
    print('MATRIX_ROUND', d.get('roundId'), d.get('updatedAt'), 'frozen', d.get('frozenAgentCount'))
    for name, row in (d.get('chains') or {}).items():
        if isinstance(row, dict):
            keep = {k: row.get(k) for k in ('generation','games','ppoWins','ppoLosses','ppoWinRate','bcWins','bcLosses','bcWinRate','seat0WinRate','seat1WinRate','seatGap','ppoVsBcWins','ppoVsBcLosses') if k in row}
            print('MATRIX_CHAIN', name, keep)
            print('MATRIX_CHAIN_KEYS', name, sorted(row))
            for key in ('frozenAggregate','universalBcFrozenAggregate','directVsUniversalBc','seatMetrics','deltaVsPrevious'):
                if key in row:
                    print('MATRIX_CHAIN_SECTION', name, key, json.dumps(row[key], separators=(',', ':'))[:1500])
PY
echo MAXBELT_FILES
find /dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-deck-branches-20260812/a08_maxbelt -maxdepth 4 -type f \( -name 'latest.json' -o -name 'league.json' -o -name 'progress.jsonl' -o -name 'latest.pt' \) -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p %s\n' 2>/dev/null | sort | tail -n 12
echo RELATED_PROCS
pgrep -af 'lucario_gold_exact|a08_maxbelt' | head -n 30 || true
