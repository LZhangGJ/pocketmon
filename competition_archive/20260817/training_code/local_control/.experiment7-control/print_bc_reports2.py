import json
from pathlib import Path
d=json.loads((Path(__file__).parent/'bc-training-now2.json').read_text(encoding='utf-8'))
for k,v in d['rolling'].items():
    if not k.endswith('training_report.json'): continue
    print(k)
    print('best',v.get('best'))
    for e in v.get('epochs',[]):
        va=e.get('validation',{})
        tr=e.get('training',{})
        print('epoch',e.get('epoch'),'nll',tr.get('policyNll'),'semantic',va.get('exactSemantic'),'index',va.get('exactIndex'),'count',va.get('countAccuracy'),'illegal',va.get('illegalPredictionCount'),'brier',va.get('valueBrier'))
