"""Offline: true-class centroids of the answer-position residual at layers 14-20 (train 0-1023).

Stored float16 and only for the pre-binding layers to keep the Kaggle bundle under its source-size cap.
"""
import numpy as np

from common import CACHE, N_CLASSES, WORK, load_meta

FIRST_LAYER, LAST_LAYER = 14, 20

hidden = np.load(CACHE / "hidden.npy").astype(np.float32)
_, true_cls = load_meta("train", range(len(hidden)))
centroids = np.stack([hidden[true_cls == c, FIRST_LAYER:LAST_LAYER + 1].mean(0) for c in range(N_CLASSES)], 1)
np.save(WORK / "centroids_true.npy", centroids.astype(np.float16))
print(centroids.shape)
