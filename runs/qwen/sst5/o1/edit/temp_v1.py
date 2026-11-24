"""Logit temperature: the base model is ~5x overconfident on the letter readout (dev NLL 3.44).
Dividing the final RMSNorm gain by T divides every logit by T (tied embeddings are left untouched).
T = 5.75 was the NLL-minimising value on 2048 base-model train outputs."""
TEMPERATURE = 5.75


def apply(model, tok):
    model.model.norm.weight.mul_(1.0 / TEMPERATURE)
