"""Token-level segmentation of a rendered prompt: persona / facts / question / per-option parts / tail."""
import numpy as np

SEG_PRE, SEG_PERSONA, SEG_FACTS, SEG_QUESTION, SEG_OPT0, SEG_TAIL = 0, 1, 2, 3, 4, 19
N_SEG = 20
N_OPT = 5
PART_LETTER, PART_CONTENT, PART_END = 0, 1, 2
N_PART = 3
PERSONA_MARK = "About the respondent:"
FACTS_MARK = "\n\nSome of their other survey answers:"
QUESTION_MARK = "\n\nQuestion:\n"
OPTIONS_MARK = "\n\nOptions:\n"
TAIL_MARK = "\n\nWhich option did this respondent choose?"
TAIL_GAP = 2


def char_segments(prompt: str) -> np.ndarray:
    seg = np.zeros(len(prompt), dtype=np.int64)
    i_opt = prompt.rfind(OPTIONS_MARK)
    i_tail = prompt.rfind(TAIL_MARK)
    i_q = prompt.rfind(QUESTION_MARK, 0, i_opt)
    i_facts = prompt.find(FACTS_MARK)
    i_persona = prompt.find(PERSONA_MARK)
    assert 0 <= i_persona < i_facts < i_q < i_opt < i_tail, "prompt layout"
    seg[i_persona:i_facts] = SEG_PERSONA
    seg[i_facts:i_q] = SEG_FACTS
    seg[i_q:i_opt + len(OPTIONS_MARK)] = SEG_QUESTION
    start = i_opt + len(OPTIONS_MARK)
    lines = prompt[start:i_tail].split("\n")
    assert len(lines) == N_OPT, lines
    for j, line in enumerate(lines):
        base = SEG_OPT0 + N_PART * j
        seg[start] = base + PART_LETTER
        seg[start + 1:start + len(line)] = base + PART_CONTENT
        end_len = TAIL_GAP if j == N_OPT - 1 else 1
        seg[start + len(line):start + len(line) + end_len] = base + PART_END
        start += len(line) + 1
    seg[i_tail + TAIL_GAP:] = SEG_TAIL
    return seg


def token_segments(tok, prompt: str) -> np.ndarray:
    """Segment id of each token, taken from the token's last character."""
    cseg = char_segments(prompt)
    offsets = tok(prompt, return_offsets_mapping=True)["offset_mapping"]
    return np.array([cseg[e - 1] if e > s else SEG_PRE for s, e in offsets], dtype=np.int64)


def option_positions(tseg: np.ndarray) -> dict:
    """Token indices per option: letter token, content tokens, last content token, end token."""
    out = {"letter": [], "content": [], "last": [], "end": []}
    for j in range(N_OPT):
        base = SEG_OPT0 + N_PART * j
        letter = np.flatnonzero(tseg == base + PART_LETTER)
        content = np.flatnonzero(tseg == base + PART_CONTENT)
        end = np.flatnonzero(tseg == base + PART_END)
        last = content[-1] if len(content) else (letter[-1] if len(letter) else -1)
        out["letter"].append(int(letter[-1]) if len(letter) else int(last))
        out["content"].append(content if len(content) else np.array([last]))
        out["last"].append(int(last))
        out["end"].append(int(end[-1]) if len(end) else int(last))
    return out
