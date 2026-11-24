"""Job 3: lookup circuit after the pad-position fix; diagnostics + full dev."""
import json
from pathlib import Path

import jobcommon as J

OUT = Path(__file__).parent / "job3"
EDIT = "lookup_v1"
DIAG_SEED = 21
CAPTURE = (0, 1, 2, 14, 27)
GO_THRESHOLD = 0.5


def main() -> None:
    lab = J.make_lab(EDIT, OUT)
    n_train = len(lab.examples("train"))
    acc = J.run_split(lab, "train", J.sample_indices(n_train, J.PASS_N, DIAG_SEED), "diag_train", OUT, CAPTURE)
    if acc >= GO_THRESHOLD:
        J.run_split(lab, "dev", list(range(len(lab.examples("dev")))), "dev_full", OUT)
    counts = J.count_changed(lab.model)
    (OUT / "changed.json").write_text(json.dumps(counts, indent=1))
    print("changed", counts["changed"], counts["fraction"], flush=True)


if __name__ == "__main__":
    main()
