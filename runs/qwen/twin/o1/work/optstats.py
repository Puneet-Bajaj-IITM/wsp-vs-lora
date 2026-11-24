"""Hook-based per-option statistics captured during charged lab.forward passes (no extra forwards)."""
import numpy as np
import torch
from wsp.scoring import render_prompts

OPTIONS_MARK = "\nOptions:\n"
LETTER_PREFIX_LEN = 3


def option_spans(tok, ex) -> tuple[list[str], list[list[int]], int]:
    """Option texts and their token indices (unpadded) in the rendered prompt."""
    prompt = render_prompts(tok, [ex])[0]
    enc = tok(prompt, return_offsets_mapping=True)
    start = prompt.rindex(OPTIONS_MARK) + len(OPTIONS_MARK)
    texts, spans, pos = [], [], start
    for _ in range(5):
        end_line = prompt.index("\n", pos)
        line = prompt[pos:end_line]
        t_start, t_end = pos + LETTER_PREFIX_LEN, end_line
        texts.append(line[LETTER_PREFIX_LEN:])
        spans.append([i for i, (s, e) in enumerate(enc["offset_mapping"])
                      if e > s and s >= t_start - 1 and e <= t_end])
        pos = end_line + 1
    return texts, spans, len(enc["input_ids"])


class OptionCapture:
    """Per example, per option: mean MLP input (post_attention_layernorm out) and mean residual (layer out)."""

    def __init__(self, lab, layers, examples, batch_size):
        self.lab, self.layers, self.bs = lab, layers, batch_size
        info = [option_spans(lab.tok, ex) for ex in examples]
        self.texts = [i[0] for i in info]
        self.spans = [i[1] for i in info]
        self.lens = [i[2] for i in info]
        n = len(examples)
        self.xin = {l: np.zeros((n, 5, 1536), np.float32) for l in layers}
        self.res = {l: np.zeros((n, 5, 1536), np.float32) for l in layers}
        self.gsum = {l: np.zeros(1536, np.float64) for l in layers}
        self.gcount = 0
        self.batch = {l: 0 for l in layers}
        self.handles = []

    def _rows(self, b):
        lo = b * self.bs
        idx = list(range(lo, min(lo + self.bs, len(self.lens))))
        maxlen = max(self.lens[i] for i in idx)
        return idx, maxlen

    def _make(self, layer, kind):
        def hook(_m, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            b = self.batch[layer] if kind == "xin" else self.batch[layer] - 1
            idx, maxlen = self._rows(b)
            store = self.xin[layer] if kind == "xin" else self.res[layer]
            for row, i in enumerate(idx):
                pad = maxlen - self.lens[i]
                for k, sp in enumerate(self.spans[i]):
                    if sp:
                        pos = torch.tensor([p + pad for p in sp], device=hs.device)
                        store[i, k] = hs[row, pos].float().mean(0).cpu().numpy()
                if kind == "xin":
                    self.gsum[layer] += hs[row, pad:].float().sum(0).double().cpu().numpy()
                    if layer == self.layers[0]:
                        self.gcount += maxlen - pad
            if kind == "xin":
                self.batch[layer] += 1
        return hook

    def __enter__(self):
        for l in self.layers:
            layer = self.lab.model.model.layers[l]
            self.handles.append(layer.post_attention_layernorm.register_forward_hook(self._make(l, "xin")))
            self.handles.append(layer.register_forward_hook(self._make(l, "res")))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
