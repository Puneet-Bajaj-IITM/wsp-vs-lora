import json, collections, numpy as np, re
D='data/processed/'
tr=[json.loads(l) for l in open(D+'train.jsonl')]
dv=[json.loads(l) for l in open(D+'dev.jsonl')]
def opts(e):
    m=e['user_message'].split('\nOptions:\n')[1].split('\n\nWhich option')[0].split('\n')
    return [x[3:] for x in m]
def q(e): return e['user_message'].split('\nQuestion:\n')[1].split('\n\nOptions:')[0]
print(len(tr),len(dv), 'cols', len({e['column'] for e in tr}), 'dev cols unseen', len({e['column'] for e in dv}-{e['column'] for e in tr}))
print('label hist', np.bincount([e['label'] for e in tr])/len(tr))
bl=collections.Counter(e['block'] for e in tr); print(bl.most_common())
print('n_item_options', collections.Counter(e['n_item_options'] for e in tr).most_common())
# option text frequency & win rate
cnt=collections.Counter(); win=collections.Counter()
for e in tr:
    o=opts(e)
    for i,t in enumerate(o):
        cnt[t]+=1; win[t]+= (i==e['label'])
print('distinct option texts',len(cnt))
for t,c in cnt.most_common(70): print(f'{c:6d} {win[t]/c:.3f} {t[:70]!r}')
# per block examples
seen=set()
for e in tr:
    if e['block'] not in seen:
        seen.add(e['block']); print('\n##',e['block'],'|',q(e)[:300],'|',opts(e),e['label'])
# char length of prompts
L=[len(e['user_message']) for e in tr]; print(np.percentile(L,[0,50,90,99,100]))
