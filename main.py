"""Pipeline lengkap: baca data/ -> preprocessing -> 5-fold CV untuk 3 model -> evaluasi -> grafik & tabel hasil.

Jalankan: python main.py                                  (semua model, pengaturan dari src/config.py)
          python main.py --epochs 1                       (uji cepat)
          python main.py --models cnn_scratch cnn_modified (hanya model tertentu)
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.config import Config
from src.data.dataset import ImageDataset, find_classes, list_images, make_folds
from src.data.preprocess import eval_transform, load_images, train_transform
from src.evaluation.metrics import METRIC_NAMES, compute_metrics, summarize
from src.evaluation.plots import plot_comparison, plot_confusion_matrix, plot_history, plot_preprocessing, plot_roc
from src.models import MODEL_NAMES, build_model
from src.training.trainer import predict, train_fold

Folds = list[tuple[np.ndarray, np.ndarray]]


def make_loader(dataset: Dataset, cfg: Config, train: bool = False) -> DataLoader:
    """DataLoader dari config. Data train diacak dan batch terakhir yang tidak penuh dibuang (aman untuk BatchNorm).

    Tanpa worker tambahan: gambar sudah ada di RAM, dan di Windows tiap worker memuat ulang torch (boros RAM).
    """
    return DataLoader(dataset, cfg.batch_size, shuffle=train, drop_last=train, pin_memory=cfg.device == "cuda")


def cross_validate(
    name: str, cfg: Config, classes: list[str], sets: tuple[Dataset, Dataset], folds: Folds, test_loader: DataLoader
) -> list[dict[str, float]]:
    """Latih dan uji satu model di setiap fold, lalu simpan grafiknya.

    Args:
        name: Nama model (lihat MODEL_NAMES).
        cfg: Konfigurasi eksperimen.
        classes: Nama kelas.
        sets: Dataset Training dengan augmentasi (untuk train) dan tanpa augmentasi (untuk valid).
        folds: Indeks train/valid setiap fold.
        test_loader: Data Testing.

    Returns:
        Metrik test setiap fold.
    """
    device = torch.device(cfg.device)
    fig_dir = cfg.output_dir / "figures" / name
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        model = build_model(name, len(classes), cfg.img_size, cfg.pretrained)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n=== {name} ({n_params:,} parameter) | fold {fold}/{len(folds)} | {device} ===")
        train_loader = make_loader(Subset(sets[0], train_idx), cfg, train=True)
        val_loader = make_loader(Subset(sets[1], val_idx), cfg)
        model, history = train_fold(model, train_loader, val_loader, cfg.epochs, cfg.lr, cfg.accum_steps, device)

        y_true, y_prob = predict(model, test_loader, device)
        metrics = compute_metrics(y_true, y_prob)
        results.append(metrics)
        print("  test  | " + " | ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

        plot_history(history, f"{name} fold {fold} - kurva training", fig_dir / f"fold{fold}_history.png")
        plot_confusion_matrix(y_true, y_prob, classes, f"{name} fold {fold}", fig_dir / f"fold{fold}_cm.png")
        plot_roc(y_true, y_prob, classes, f"{name} fold {fold} - ROC", fig_dir / f"fold{fold}_roc.png")
    return results


def save_results(all_results: dict[str, list[dict[str, float]]], out_path: Path) -> None:
    """Simpan tabel Markdown: metrik per fold tiap model + perbandingan rata-rata (siap ditempel ke README)."""
    def row(*cells: str) -> str:
        return "| " + " | ".join(cells) + " |"

    sep = "|---" * (len(METRIC_NAMES) + 1) + "|"
    lines = []
    for name, results in all_results.items():
        lines += [f"### {name}", "", row("Fold", *METRIC_NAMES), sep]
        lines += [row(str(i), *(f"{r[m]:.4f}" for m in METRIC_NAMES)) for i, r in enumerate(results, 1)]
        lines.append("")
    lines += ["### Perbandingan model (rata-rata ± std)", "", row("Model", *METRIC_NAMES), sep]
    for name, results in all_results.items():
        lines.append(row(name, *(f"{mu:.4f} ± {sd:.4f}" for mu, sd in summarize(results).values())))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config) -> dict[str, dict[str, tuple[float, float]]]:
    """Jalankan seluruh eksperimen.

    Returns:
        Rata-rata ± std setiap metrik dari semua fold, per model.
    """
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    data_dir = cfg.data_dir
    if not (data_dir / "Training").is_dir() or not (data_dir / "Testing").is_dir():
        raise FileNotFoundError(f"Ekstrak archive.zip ke {data_dir.resolve()} sehingga ada folder Training dan Testing")
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA tidak terdeteksi: install torch CUDA dari requirements.txt atau set device='cpu'")
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    classes = find_classes(data_dir / "Training")
    train_paths, train_labels = list_images(data_dir / "Training", classes)
    test_paths, test_labels = list_images(data_dir / "Testing", classes)
    print(f"Kelas: {classes} | gambar Training: {len(train_paths)} | gambar Testing: {len(test_paths)}")
    examples = [train_paths[int(np.argmax(train_labels == i))] for i in range(len(classes))]
    plot_preprocessing(examples, cfg.img_size, cfg.output_dir / "preprocessing.png")

    train_images, test_images = load_images(train_paths, cfg.img_size), load_images(test_paths, cfg.img_size)
    sets = (
        ImageDataset(train_images, train_labels, train_transform()),
        ImageDataset(train_images, train_labels, eval_transform()),
    )
    test_loader = make_loader(ImageDataset(test_images, test_labels, eval_transform()), cfg)
    folds = make_folds(train_labels, cfg.n_folds, cfg.seed)

    all_results = {name: cross_validate(name, cfg, classes, sets, folds, test_loader) for name in cfg.models}
    summaries = {name: summarize(results) for name, results in all_results.items()}
    save_results(all_results, cfg.output_dir / "results.md")
    plot_comparison(summaries, cfg.output_dir / "comparison.png")

    print(f"\nRata-rata {cfg.n_folds}-fold (lengkap di {cfg.output_dir / 'results.md'}):")
    for name, summary in summaries.items():
        print(f"  {name:16s}: " + " | ".join(f"{m} {mu:.4f}" for m, (mu, _) in summary.items()))
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Klasifikasi Brain Tumor MRI: 3 model dengan 5-fold cross validation")
    parser.add_argument("--epochs", type=int, default=Config.epochs, help="jumlah epoch per fold")
    parser.add_argument("--folds", type=int, default=Config.n_folds, help="jumlah fold cross validation")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES, help="model yang dilatih")
    args = parser.parse_args()
    run(Config(epochs=args.epochs, n_folds=args.folds, models=tuple(args.models)))