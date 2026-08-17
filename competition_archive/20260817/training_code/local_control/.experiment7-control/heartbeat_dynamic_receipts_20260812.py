#!/usr/bin/env python3
import json
from pathlib import Path

root = Path('/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811')

def load(p):
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        return {'error': str(exc), 'path': str(p)}

league = load(root / 'state/league.json')
state = load(root / 'state/adaptive-training-state.json')
out = {
    'leagueUpdatedAt': league.get('trainingControlUpdatedAt'),
    'sourceRoundId': league.get('trainingControlSourceRoundId'),
    'state': {k: state.get(k) for k in ('lastRoundId', 'lastAppliedAt')},
    'chains': {},
}
for name, chain in league.get('chains', {}).items():
    summaries = sorted(root.glob(f'buffer/**/{name}/**/*.summary.json'), key=lambda p: p.stat().st_mtime)
    batches = sorted(root.glob(f'learners/{name}/**/batch.json'), key=lambda p: p.stat().st_mtime)
    summary = None
    summary_path = None
    for candidate in reversed(summaries):
        payload = load(candidate)
        sampling = payload.get('samplingControl') if isinstance(payload, dict) else None
        if isinstance(sampling, dict) and sampling.get('agentWeights'):
            summary = payload
            summary_path = candidate
            break
    if summary is None and summaries:
        summary = load(summaries[-1])
        summary_path = summaries[-1]
    batch = load(batches[-1]) if batches else None
    out['chains'][name] = {
        'generation': chain.get('generation'),
        'control': chain.get('trainingControl'),
        'latestSummaryPath': str(summary_path) if summary_path else None,
        'latestSummaryMtime': summary_path.stat().st_mtime if summary_path else None,
        'latestSamplingControl': summary.get('samplingControl') if isinstance(summary, dict) else None,
        'latestSummaryEpisodes': summary.get('episodes') if isinstance(summary, dict) else None,
        'latestSummaryDecisions': summary.get('decisions') if isinstance(summary, dict) else None,
        'latestBatchPath': str(batches[-1]) if batches else None,
        'latestBatchMtime': batches[-1].stat().st_mtime if batches else None,
        'latestBatchControl': batch.get('trainingControl') if isinstance(batch, dict) else None,
        'latestBatchDecisions': batch.get('decisions') if isinstance(batch, dict) else None,
    }
print(json.dumps(out, ensure_ascii=False))
