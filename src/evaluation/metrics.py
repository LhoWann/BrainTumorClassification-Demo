"""Metrik evaluasi: accuracy, precision, recall, F1, ROC-AUC, dan rata-rata antar fold."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Hitung metrik klasifikasi multi-kelas (rata-rata macro, ROC-AUC one-vs-rest).

    Args:
        y_true: Label asli (N,).
        y_prob: Probabilitas prediksi (N, jumlah_kelas).

    Returns:
        Nilai setiap metrik.
    """
    y_pred = y_prob.argmax(1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"),
    }


def summarize(fold_results: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    """Rata-rata dan standar deviasi setiap metrik dari semua fold."""
    values = {m: [r[m] for r in fold_results] for m in METRIC_NAMES}
    return {m: (float(np.mean(v)), float(np.std(v))) for m, v in values.items()}