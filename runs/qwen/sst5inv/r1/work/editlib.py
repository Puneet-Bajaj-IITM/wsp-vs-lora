"""Weight-edit primitives used while screening variants (the final edit file is self-contained)."""
import torch

NEGATIVE, POSITIVE, NEUTRAL, VERY = " negative", " positive", " neutral", " very"


def token_id(tok, text: str) -> int:
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) == 1, (text, ids)
    return ids[0]


def is_single_token(tok, text: str) -> bool:
    return len(tok.encode(text, add_special_tokens=False)) == 1


def embedding(model):
    return model.get_input_embeddings().weight


def swap_rows(model, tok, a: str = NEGATIVE, b: str = POSITIVE) -> None:
    weight, ia, ib = embedding(model), token_id(tok, a), token_id(tok, b)
    row = weight[ia].clone()
    weight[ia] = weight[ib]
    weight[ib] = row


def blend_row(model, tok, target: str, sources: list[str], mix: float, original_rows: dict) -> None:
    """target row <- (1-mix)*target + mix*mean(sources), rescaled to the target's original norm."""
    weight = embedding(model)
    base = original_rows[target].float()
    donor = torch.stack([original_rows[s].float() for s in sources]).mean(0)
    donor = donor * base.norm() / donor.norm()
    new = (1 - mix) * base + mix * donor
    weight[token_id(tok, target)] = (new * base.norm() / new.norm()).to(weight.dtype)


def rank_one_steer(model, layer: int, steer: torch.Tensor, key: torch.Tensor, top_k: int) -> int:
    """down_proj[layer] += steer (x) key_k / |key_k|^2, key_k = key restricted to its top_k largest entries.

    Adds ~steer to the block output wherever the MLP activation resembles the mean answer-position activation."""
    weight = model.model.layers[layer].mlp.down_proj.weight
    key = key.to(weight.device).float()
    cols = key.abs().topk(top_k).indices
    key_k = key[cols]
    update = torch.outer(steer.to(weight.device).float(), key_k / key_k.pow(2).sum())
    weight[:, cols] = (weight[:, cols].float() + update).to(weight.dtype)
    return update.numel()
