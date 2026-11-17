"""Gambar hasil: contoh preprocessing, kurva training, confusion matrix, dan kurva ROC."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.data.preprocess import CropBlackBorder


def plot_preprocessing(paths: list[Path], img_size: int, out_path: Path) -> None:
    """Bandingkan gambar asli dan gambar setelah crop + resize."""
    fig, axes = plt.subplots(2, len(paths), figsize=(3 * len(paths), 6), squeeze=False)
    for col, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        processed = CropBlackBorder()(image).resize((img_size, img_size))
        for row, (img, title) in enumerate(((image, "asli"), (processed, "preprocessed"))):
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"{path.parent.name} ({title})", fontsize=8)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_history(history: dict[str, list[float]], title: str, out_path: Path) -> None:
    """Kurva loss dan akurasi train vs valid per epoch."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    for ax, key in ((ax_loss, "loss"), (ax_acc, "acc")):
        ax.plot(epochs, history[f"train_{key}"], marker="o", label="train")
        ax.plot(epochs, history[f"val_{key}"], marker="o", label="valid")
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, classes: list[str], title: str, out: Path) -> None:
    """Confusion matrix dari data test."""
    fig, ax = plt.subplots(figsize=(6, 5))
    y_pred = y_prob.argmax(1)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=classes, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_comparison(summaries: dict[str, dict[str, tuple[float, float]]], out_path: Path) -> None:
    """Diagram batang rata-rata ± std setiap metrik untuk semua model."""
    metrics = list(next(iter(summaries.values())))
    x, width = np.arange(len(metrics)), 0.8 / len(summaries)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (name, summary) in enumerate(summaries.items()):
        means, stds = zip(*(summary[m] for m in metrics), strict=True)
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=name)
    ax.set_xticks(x + width * (len(summaries) - 1) / 2, metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rata-rata 5-fold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, classes: list[str], title: str, out_path: Path) -> None:
    """Kurva ROC one-vs-rest untuk setiap kelas."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, name in enumerate(classes):
        RocCurveDisplay.from_predictions(y_true == i, y_prob[:, i], name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)