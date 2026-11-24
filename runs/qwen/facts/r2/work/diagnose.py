"""Decode the designed codes from captured answer-position residuals (simple projections)."""
import numpy as np

ATTR_INDEX = {"city": 0, "profession": 1, "hobby": 2}
NUM_VALUES = 10


def diagnose(hidden, examples, design, select_layer=1):
    """hidden: {layer: (n, d)}; design: _hj_build.LAST. Returns a dict of stage accuracies."""
    codes, plan, cfg = design["codes"], design["plan"], design["cfg"]
    report = {}
    pid = np.array([e["pid"] for e in examples])
    attr = np.array([ATTR_INDEX[e["column"]] for e in examples])
    labels = np.array([e["label"] for e in examples])
    truth = plan.table[pid, attr]
    if 0 in hidden:
        h0 = hidden[0].astype(np.float64)
        report["l0_norm"] = float(np.linalg.norm(h0, axis=1).mean())
        value = h0 @ codes.rows("value").numpy().T / cfg.code_scale
        block = np.stack([value[i, a * NUM_VALUES:(a + 1) * NUM_VALUES] for i, a in enumerate(attr)])
        report["l0_value_code_acc"] = float((block.argmax(1) == truth).mean())
        report["l0_true_code_mean"] = float(block[np.arange(len(truth)), truth].mean())
        report["l0_other_code_absmax_mean"] = float(np.abs(np.where(np.eye(NUM_VALUES)[truth] > 0, 0, block)).max(1).mean())
        bag = h0 @ codes.rows("name").numpy().T
        sink = h0 @ codes.rows("sink").numpy().T
        est = 3 * bag / np.maximum(sink, 1e-6)
        report["l0_bag_abs_err_mean"] = float(np.abs(est - plan.bags[pid]).max(1).mean())
        report["l0_sink_coef_mean"] = float(sink.mean() / cfg.code_scale)
    for layer in sorted(k for k in hidden if k != 0):
        h = hidden[layer].astype(np.float64)
        proxy = h @ design["readout"].numpy().T / np.sqrt((h ** 2).mean(1, keepdims=True))
        report[f"l{layer}_norm"] = float(np.linalg.norm(h, axis=1).mean())
        report[f"l{layer}_readout_acc"] = float((proxy.argmax(1) == labels).mean())
        srt = np.sort(proxy, axis=1)
        report[f"l{layer}_readout_gap"] = float((srt[:, -1] - srt[:, -2]).mean())
    return report
