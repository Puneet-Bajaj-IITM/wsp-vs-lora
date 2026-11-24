import json, collections, numpy as np
D='data/processed/'
tr=[json.loads(l) for l in open(D+'train.jsonl')]
dv=[json.loads(l) for l in open(D+'dev.jsonl')]
print(len(tr),len(dv))
cols=collections.Counter(e['column'] for e in tr)
print('items',len(cols),'min/median/max per item',min(cols.values()),np.median(list(cols.values())),max(cols.values()))
print('blocks',collections.Counter(e['block'] for e in tr).most_common())
print('n_item_options',collections.Counter(e['n_item_options'] for e in tr).most_common())
print('label hist',np.bincount([e['label'] for e in tr])/len(tr), np.bincount([e['label'] for e in dv])/len(dv))
def opts(e):
    ls=e['user_message'].split('Options:\n')[-1].split('\n\nWhich option')[0].split('\n')
    return [l[3:] for l in ls]
# priors
item=collections.defaultdict(lambda: collections.Counter())
txt=collections.Counter(); txts=collections.Counter()
btxt=collections.Counter(); btxts=collections.Counter()
for e in tr:
    o=opts(e); assert len(o)==5,o
    item[e['column']][e['shown_option_ids'][e['label']]]+=1
    for i,t in enumerate(o):
        txts[t]+=1; btxts[(e['block'],t)]+=1
        if i==e['label']: txt[t]+=1; btxt[(e['block'],t)]+=1
def ev(score):
    acc=[];
    for e in dv:
        s=score(e); acc.append(int(np.argmax(s))==e['label'])
    return np.mean(acc)
print('item prior acc',ev(lambda e:[item[e['column']][i]+0.01*np.random.rand() for i in e['shown_option_ids']]))
print('text prior acc',ev(lambda e:[(txt[t]+.5)/(txts[t]+2.5) for t in opts(e)]))
print('block-text prior acc',ev(lambda e:[(btxt[(e['block'],t)]+.5)/(btxts[(e['block'],t)]+2.5) for t in opts(e)]))
print('distinct option texts',len(txts))
# per block: item prior acc, n
bl=collections.defaultdict(list)
for e in dv:
    s=[item[e['column']][i] for i in e['shown_option_ids']]
    bl[e['block']].append(int(np.argmax(s))==e['label'])
for b,v in sorted(bl.items(),key=lambda kv:-len(kv[1])): print(b,len(v),round(np.mean(v),3), len({e['column'] for e in tr if e['block']==b}))
print(txts.most_common(40))
