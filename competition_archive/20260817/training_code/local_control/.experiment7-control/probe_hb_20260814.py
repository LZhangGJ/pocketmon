import json
import os
import subprocess
from pathlib import Path

MAIN = Path('/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811')
chains = [
    'a02_grim_g247', 'a02_grim_g247_pokegear', 'a08_rabsca',
    'a08_maxbelt', 'lucario_gold_exact', 'universal_ppo_standard_1m',
    'universal_ppo_large_256x6',
]
raw = subprocess.check_output([
    '/homes/lzhang/mypath/new/envs/trans/bin/python',
    '/homes/lzhang/summarize_async_ppo_league.py',
], text=True, timeout=180)
summary = json.loads(raw)
out = {'leagueUpdatedAt': summary.get('leagueUpdatedAt'), 'chains': {}}
for name in chains:
    row = summary['chains'][name]
    out['chains'][name] = {
        key: row.get(key) for key in (
            'generation', 'snapshotId', 'completedShards', 'episodes', 'decisions',
            'externalWins', 'externalLosses', 'selfPlayEpisodes',
            'livePpoOpponentEpisodes', 'publishedUpdates', 'failedUpdates',
            'latestInitialPolicyShift', 'latestEpoch',
        )
    }
for rel, key in [
    ('monitoring/ppo-vs-bc-tier-a/latest.json', 'tierA'),
    ('monitoring/specialist-loss-replays/latest.json', 'lossArchive'),
    ('monitoring/specialist-loss-replays/controller-state.json', 'lossArchiveController'),
    ('monitoring/gold-acceleration/latest.json', 'previousSnapshot'),
]:
    path = MAIN / rel
    if path.exists():
        try:
            out[key] = json.loads(path.read_text())
        except Exception as exc:
            out[key] = {'error': repr(exc), 'path': str(path)}
matrix = sorted(MAIN.glob('monitoring/**/*report.json'), key=lambda p: p.stat().st_mtime, reverse=True)
out['latestReports'] = [
    {'path': str(p), 'mtime': p.stat().st_mtime, 'size': p.stat().st_size}
    for p in matrix[:10]
]
print(json.dumps(out))
