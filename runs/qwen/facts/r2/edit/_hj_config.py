"""Constants for the closed-form lookup circuit."""
from dataclasses import dataclass

ATTRS = ("city", "profession", "hobby")
NUM_VALUES = 10
SINK_COUNT = 3  # <|im_start|> tokens visible from the answer position

# layer-0 head roles (query heads 0-5 share kv group 0, heads 6-11 share kv group 1)
HEAD_NAME = 6
HEAD_LETTER = 7
HEAD_SELF = 0
HEAD_SELECT = 0
HEADS_PER_GROUP = 6

# rotary pair indices (pair i couples head dims i and i + head_dim // 2)
PAIR_NAME_TARGET = 63
PAIR_SINK = 62
KERNEL_PAIRS = tuple(range(12))
SELECT_PAIRS = tuple(range(49, 64))
PAIR_SELECT_SINK = 48

LETTER_OFFSET_BASE = 2
MAX_KEPT_NAME_TOKENS = 120
MIN_BAG_DISTANCE = 2
KERNEL_CHECK_RANGE = range(3, 90)
KERNEL_MAX_SIDELOBE = 0.7
CODE_SEED = 1234
NEUTRAL_SEED = 4321


@dataclass(frozen=True)
class HijackConfig:
    code_scale: float = 300.0
    write_scale: float = 4000.0
    logit_gap: float = 6.0
    beta_name: float = 9.0
    beta_sink: float = 12.0
    beta_letter: float = 30.0
    beta_self: float = 40.0
    beta_select: float = 30.0
    gate_on: float = 6.0
    select_layer: int = 1
    letter_bias: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)
