import json, collections, numpy as np
D='data/processed/'
tr=[json.loads(l) for l in open(D+'train.jsonl')]
dv=[json.loads(l) for l in open(D+'dev.jsonl')]
def opts(e):
    m=e['user_message'].split('\nOptions:\n')[1].split('\n\nWhich option')[0].split('\n')
    return [x[3:] for x in m]
def table(keyf):
    c=collections.Counter(); w=collections.Counter()
    for e in tr:
        for i,t in enumerate(opts(e)):
            k=keyf(e,t); c[k]+=1; w[k]+=(i==e['label'])
    return c,w
def ev(keyf,name):
    c,w=table(keyf); acc=[];nll=[]; byb=collections.defaultdict(list)
    for e in dv:
        s=np.array([(w[keyf(e,t)]+1)/(c[keyf(e,t)]+5) for t in opts(e)]); p=s/s.sum()
        ok=p.argmax()==e['label']; acc.append(ok); nll.append(-np.log(p[e['label']])); byb[e['block']].append(ok)
    print(name, round(np.mean(acc),4), round(np.mean(nll),4), {b[:12]:round(np.mean(v),3) for b,v in byb.items() if len(v)>=50})
ev(lambda e,t:t,'global text')
ev(lambda e,t:(e['block'],t),'block text')
ev(lambda e,t:(e['column'],t),'item text')
print(collections.Counter(e['block'] for e in dv).most_common(6))
# cognitive items
cog=collections.defaultdict(collections.Counter)
qs={}
for e in tr:
    if e['block']=='Cognitive tests':
        o=opts(e); cog[e['column']][o[e['label']]]+=1; qs[e['column']]=e['user_message'].split('\nQuestion:\n')[1].split('\n\nOptions:')[0][:110]
for k,v in cog.items(): print(k, qs[k],'|', v.most_common(3), sum(v.values()))
eco=collections.defaultdict(collections.Counter)
for e in tr:
    if e['block'] in('Economic preferences',):
        o=opts(e); eco[e['column']][o[e['label']]]+=1; qs[e['column']]=e['user_message'].split('\nQuestion:\n')[1].split('\n\nOptions:')[0][:110]
for k,v in eco.items(): print(k, qs[k],'|', v.most_common(2), sum(v.values()))
