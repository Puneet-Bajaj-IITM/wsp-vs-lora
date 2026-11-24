"""Unit test of the circuit construction on a RANDOM-WEIGHT stand-in network (never the real Qwen weights)."""
import importlib.util, json, math, sys
import numpy as np, torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
sys.path.insert(0, "src")
from wsp.prompts import build_chat

torch.manual_seed(1)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct", local_files_only=True)
tok.padding_side = "left"
cfg = Qwen2Config(vocab_size=151936, hidden_size=1536, intermediate_size=8960, num_hidden_layers=int(sys.argv[1]) if len(sys.argv) > 1 else 3,
                  num_attention_heads=12, num_key_value_heads=2, rope_theta=1e6, tie_word_embeddings=True,
                  attn_implementation="eager")
model = Qwen2ForCausalLM(cfg).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("bias"): p.normal_(0, 0.3)
        if "norm" in name or "layernorm" in name: p.uniform_(0.5, 2.0)
        if "mlp.down" in name or "o_proj" in name: p.mul_(float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
spec = importlib.util.spec_from_file_location("circuit", "edits/facts/r4/circuit.py")
circuit = importlib.util.module_from_spec(spec); spec.loader.exec_module(circuit)
train = [json.loads(l) for l in open("data/processed/facts/train.jsonl")]
dev = [json.loads(l) for l in open("data/processed/facts/dev.jsonl")]
rng = np.random.default_rng(0)
NEX = int(sys.argv[4]) if len(sys.argv) > 4 else 24
exs = [train[i] for i in rng.choice(len(train), NEX, replace=False)] + [dev[i] for i in rng.choice(len(dev), NEX, replace=False)]
prompts = [tok.apply_chat_template(build_chat(e["user_message"], e["system"]), tokenize=False, add_generation_prompt=True) for e in exs]
letters = [tok.encode(l, add_special_tokens=False)[0] for l in "ABCDE"]
labels = np.array([e["label"] for e in exs])

@torch.no_grad()
def run():
    batch = tok(prompts, return_tensors="pt", padding=True)
    out = model(**batch, output_hidden_states=True, output_attentions=True)
    return batch, out

batch, base = run()
n0 = base.hidden_states[1][:, -1].norm(dim=-1).median().item()
nl = base.hidden_states[-1][:, -1].norm(dim=-1).median().item()
print("base norms layer0 %.2f last %.2f  emb %.3f" % (n0, nl, model.model.embed_tokens.weight[:2000].detach().norm(dim=-1).median()))
HALF = len(sys.argv) > 3 and sys.argv[3] == "half"
if HALF: model.half()
des = circuit.apply(model, tok, scale=30 * n0, omega=20 * 30 * n0)
print("m", des.m, "sv range", des.sv[0].item(), des.sv[-1].item(), "channel ratios", des.channel_ratio)
print("rival name max %.3f null max %.3f | weights changed %d | rho %.4f" % (des.rival_name.max(), des.null_response.abs().max(), des.weights_changed, des.rho))
batch, out = run()
logp = torch.log_softmax(out.logits[:, -1, letters].double(), -1).numpy()
print("acc", (logp.argmax(1) == labels).mean(), "nll", -logp[np.arange(len(labels)), labels].mean())
print("pred", logp.argmax(1), "\nlab ", labels)
# mechanism checks on example 0..3
ids = batch["input_ids"]
for b in range(3):
    seq = ids[b].tolist(); L = len(seq)
    e = exs[b]
    val_tok = des.value_tokens[["city", "profession", "hobby"].index(e["column"])][e["shown_option_ids"][e["label"]]]
    p = max(i for i, t in enumerate(seq) if t == val_tok)
    a0 = out.attentions[0][b]; a1 = out.attentions[1][b]
    pieces = set(des.name_pieces[e["pid"]])
    name_pos = [i for i, t in enumerate(seq) if t in pieces and i < p and i > 20]
    print(f"ex{b}: p={p} tok={tok.decode([seq[p]])!r} label={e['label']}")
    for h in circuit.GATHER_HEADS:
        print("  gather head", h, "mass on name pieces at p: %.4f" % a0[h, p, name_pos].sum().item(), "per piece", a0[h, p, name_pos].numpy().round(3))
    print("  bind head attn at p to p-2: %.4f (token %r)" % (a0[circuit.BIND_HEAD, p, p - 2].item(), tok.decode([seq[p - 2]])))
    print("  read head attn from last to p: %.4f" % a1[circuit.READ_HEAD, -1, p].item())
    a0, a1 = a0.float(), a1.float()
    h0 = out.hidden_states[1][b].double()
    print("  flag coordinate at p: %.1f  elsewhere max: %.1f  |h0[p]| %.1f" % (h0[p] @ des.psi_hat, (h0 @ des.psi_hat)[[i for i in range(L) if i != p]].abs().max(), h0[p].norm()))
    hl = out.hidden_states[2][b, -1].double()
    print("  final-pos after L1: cos with answer dirs", [round(float(torch.nn.functional.cosine_similarity(hl, d, dim=0)), 3) for d in des.answer_dirs])
    nrm = h0.norm(dim=-1); proj = (h0 @ des.psi_hat)
    top = torch.argsort(-(proj / nrm))[:4]
    print("  top normalised flag positions:", [(int(i), tok.decode([seq[i]]), round(float(proj[i] / nrm[i]), 3), round(float(nrm[i]))) for i in top])
    big = torch.argsort(-proj.abs())[:3]
    print("  largest raw flag coords:", [(int(i), tok.decode([seq[i]]), round(float(proj[i])), round(float(nrm[i]))) for i in big])
    print("  bind attn at last pos top:", [(int(i), tok.decode([seq[i]]), round(float(a0[circuit.BIND_HEAD, -1, i]), 3)) for i in torch.argsort(-a0[circuit.BIND_HEAD, -1])[:3]])
    q = p + 1
    print("  bind attn at p+1 top:", [(int(i), tok.decode([seq[i]]), round(float(a0[circuit.BIND_HEAD, q, i]), 3)) for i in torch.argsort(-a0[circuit.BIND_HEAD, q])[:3]])
    q = [i for i, t in enumerate(seq) if tok.decode([t]) == "Question"][0]
    print("  gather attn at 'Question' top:", [(int(i), tok.decode([seq[i]]), round(float(a0[0, q, i]), 3)) for i in torch.argsort(-a0[0, q])[:3]])

# exercise the lab helpers on the stand-in
sys.path.insert(0, "wsp_work/facts_r4")
import labtools as lt
rep = lt.design_report(des, circuit)
print({k: v for k, v in rep.items()})
hid = {0: out.hidden_states[1][:, -1].float().numpy(), 1: out.hidden_states[2][:, -1].float().numpy(),
       -1: model.model.norm(out.hidden_states[-1][:, -1]).float().detach().numpy()}
fake = {"examples": exs, "hidden": hid, "labels": labels, "logp": logp}
print(lt.final_position_checks(des, fake))
print(lt.errors(fake)); print("headroom", lt.detector_headroom(des, fake))
circuit.revert(des)
_, back = run()
print("revert restores base:", torch.allclose(back.logits[:, -1], base.logits[:, -1]))
