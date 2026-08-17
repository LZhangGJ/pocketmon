import csv
import glob
import json
import sys
from collections import defaultdict

root = sys.argv[1] if len(sys.argv) > 1 else '/suedata1/Free/lzhang/pocketmon-runs/experiment7-hourly-cache-pool-distributed/monitoring/hourly-cache-pool/rounds/20260814T103053Z-allppo-11-2bf5d38f26e96c64/raw'
stats = defaultdict(lambda: {'games': 0, 'failures': 0, 'fallback': 0, 'truncated': 0})
failure_rows = []
for path in glob.glob(root + '/results-shard-*.csv'):
    with open(path, encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            learner = row['learner']
            if not learner.startswith('live_'):
                continue
            s = stats[learner]
            s['games'] += 1
            s['failures'] += bool(row.get('failure'))
            if row.get('failure'):
                failure_rows.append({k: row.get(k) for k in ('learner', 'opponent', 'seed', 'learner_seat', 'result', 'failure')})
            try:
                diag = json.loads(row.get('diagnostics_json') or '[]')
                own = diag[int(row['learner_seat'])].get('bc', {})
                s['fallback'] += int(own.get('fallbackCalls', 0))
                s['truncated'] += int(own.get('truncatedEntityCalls', 0))
            except (ValueError, IndexError, KeyError, TypeError):
                pass
print(json.dumps(stats, sort_keys=True))
print(json.dumps(failure_rows, sort_keys=True))
