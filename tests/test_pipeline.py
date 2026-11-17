"""Tes cepat seluruh komponen memakai dataset sintetis kecil (tanpa GPU)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from main import run
from src.config import Config
from src.data.dataset import ImageDataset, find_classes, list_images, make_folds
from src.data.preprocess import CropBlackBorder, eval_transform, load_images, train_transform
from src.evaluation.metrics import METRIC_NAMES, compute_metrics, summarize
from src.models import MODEL_NAMES, build_model
from src.training.trainer import run_epoch

CLASSES = ["glioma", "notumor", "pituitary"]


def make_image(seed: int) -> Image.Image:
    """Gambar acak 48x48 dengan tepi hitam 8 piksel."""
    canvas = np.zeros((48, 48, 3), dtype=np.uint8)
    canvas[8:40, 8:40] = np.random.default_rng(seed).integers(50, 255, (32, 32, 3), dtype=np.uint8)
    return Image.fromarray(canvas)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    for split, n in (("Training", 6), ("Testing", 3)):
        for c, name in enumerate(CLASSES):
            folder = tmp_path / "data" / split / name
            folder.mkdir(parents=True)
            for i in range(n):
                make_image(c * 100 + i).save(folder / f"{split}_{i}.png")
    return Config(
        data_dir=tmp_path / "data", output_dir=tmp_path / "out", img_size=32, batch_size=4,
        accum_steps=2, epochs=1, n_folds=2, pretrained=False, device="cpu",
    )


def test_list_images(cfg: Config) -> None:
    classes = find_classes(cfg.data_dir / "Training")
    paths, labels = list_images(cfg.data_dir / "Training", classes)
    if classes != CLASSES or len(paths) != 18 or np.bincount(labels).tolist() != [6, 6, 6]:
        raise AssertionError(f"scan salah: {classes}, {len(paths)}, {np.bincount(labels)}")


def test_make_folds_stratified() -> None:
    labels = np.repeat([0, 1, 2], 10)
    folds = make_folds(labels, n_folds=5, seed=0)
    all_val = np.concatenate([val for _, val in folds])
    if len(folds) != 5 or sorted(all_val.tolist()) != list(range(30)):
        raise AssertionError("setiap sampel harus muncul tepat sekali sebagai data valid")
    for train_idx, val_idx in folds:
        if set(train_idx) & set(val_idx) or np.bincount(labels[val_idx]).tolist() != [2, 2, 2]:
            raise AssertionError("fold bocor atau tidak stratified")


def test_crop_black_border() -> None:
    if CropBlackBorder()(make_image(0)).size != (32, 32):
        raise AssertionError("tepi hitam tidak terpotong")
    black = Image.new("RGB", (20, 20))
    if CropBlackBorder()(black).size != (20, 20):
        raise AssertionError("gambar hitam penuh harus dikembalikan utuh")


def test_load_images_and_dataset(cfg: Config) -> None:
    paths, labels = list_images(cfg.data_dir / "Training", CLASSES)
    images = load_images(paths, 24)
    if len(images) != 18 or {(im.size, im.mode) for im in images} != {((24, 24), "RGB")}:
        raise AssertionError("load_images harus menghasilkan gambar RGB 24x24")
    for transform in (train_transform(), eval_transform()):
        image, label = ImageDataset(images, labels, transform)[0]
        if image.shape != (3, 24, 24) or label != 0:
            raise AssertionError(f"output dataset salah: {image.shape}, {label}")


def test_grad_accumulation_equals_big_batch() -> None:
    data = TensorDataset(torch.randn(8, 3, 32, 32), torch.arange(8) % 3)
    weights = []
    for batch_size, accum in ((8, 1), (4, 2)):
        torch.manual_seed(0)
        model = build_model("cnn_scratch", num_classes=3, img_size=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        run_epoch(model, DataLoader(data, batch_size), nn.CrossEntropyLoss(), torch.device("cpu"), optimizer, accum)
        weights.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
    if not torch.allclose(weights[0], weights[1], atol=1e-5):
        raise AssertionError("batch 4 x accum 2 harus sama dengan batch 8 x accum 1")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_build_model(name: str) -> None:
    logits = build_model(name, num_classes=3, img_size=32, pretrained=False)(torch.randn(2, 3, 32, 32))
    if logits.shape != (2, 3):
        raise AssertionError(f"shape logits {name} salah: {logits.shape}")


def test_build_model_unknown() -> None:
    with pytest.raises(ValueError, match="tidak dikenal"):
        build_model("resnet25", num_classes=3, img_size=32)


def test_metrics() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    perfect = compute_metrics(y_true, np.eye(3)[y_true])
    if any(abs(perfect[m] - 1.0) > 1e-9 for m in METRIC_NAMES):
        raise AssertionError(f"prediksi sempurna harus bernilai 1: {perfect}")
    summary = summarize([{m: 0.5 for m in METRIC_NAMES}, {m: 1.0 for m in METRIC_NAMES}])
    if summary["f1"] != (0.75, 0.25):
        raise AssertionError(f"rata-rata/std salah: {summary['f1']}")


def test_missing_data_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="archive.zip"):
        run(Config(data_dir=tmp_path / "kosong", output_dir=tmp_path / "out"))


def test_cuda_required(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg.device = "cuda"
    with pytest.raises(RuntimeError, match="CUDA tidak terdeteksi"):
        run(cfg)


def test_run_end_to_end(cfg: Config) -> None:
    summaries = run(cfg)
    if set(summaries) != set(MODEL_NAMES) or any(set(s) != set(METRIC_NAMES) for s in summaries.values()):
        raise AssertionError(f"hasil kurang: {summaries}")
    names = ["preprocessing.png", "comparison.png", "results.md"]
    names += [f"figures/{m}/fold{k}_{p}.png" for m in MODEL_NAMES for k in (1, 2) for p in ("history", "cm", "roc")]
    missing = [n for n in names if not (cfg.output_dir / n).exists()]
    if missing:
        raise AssertionError(f"output tidak lengkap: {missing}")