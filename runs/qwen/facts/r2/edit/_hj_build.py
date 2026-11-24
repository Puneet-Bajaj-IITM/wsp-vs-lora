"""Entry point: assemble the lookup circuit. LAST keeps design tensors for diagnostics."""
from _hj_config import CODE_SEED, HijackConfig
from _hj_layer0 import build_layer0
from _hj_select import build_select, letter_readout
from _hj_tokens import plan_tokens
from _hj_writer import Codes, Writer

LAST = {}


def install(model, tok, cfg: HijackConfig = HijackConfig(), log=print):
    plan = plan_tokens(tok)
    codes = Codes(len(plan.kept), model.config.hidden_size, letter_readout(model, plan.letters), CODE_SEED)
    writer = Writer()
    neurons = build_layer0(model, plan, codes, cfg, writer, log)
    select = build_select(model, plan, codes, cfg, writer, log)
    changed, total = writer.changed(), sum(p.numel() for p in model.parameters())
    log(f"weights changed: {changed} of {total} ({changed / total:.3e})")
    LAST.update(plan=plan, codes=codes, cfg=cfg, neurons=neurons, changed=changed, total=total, **select)
