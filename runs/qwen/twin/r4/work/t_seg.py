import sys, json, glob, os
sys.path.insert(0,'src'); sys.path.insert(0,'wsp_work/twin_r4')
from transformers import AutoTokenizer
from wsp.scoring import render_prompts
import segs, numpy as np, collections
p=glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/*/'))[0]
tok=AutoTokenizer.from_pretrained(p)
tr=[json.loads(l) for l in open('data/processed/train.jsonl')][:3000:7]
ps=render_prompts(tok,tr)
lens=[];miss=collections.Counter()
for k,pr in enumerate(ps):
    ts=segs.token_segments(tok,pr); lens.append(len(ts))
    for s in range(4,19):
        if not (ts==s).any(): miss[(s-4)%3]+=1
    if k<2:
        ids=tok(pr)['input_ids']
        print([(tok.decode([i]),int(s)) for i,s in zip(ids,ts)][-70:])
print('len',min(lens),np.mean(lens),max(lens),'missing parts',miss, 'n',len(ps))
print(tok.encode('A'),tok.encode('\nA. x'))
