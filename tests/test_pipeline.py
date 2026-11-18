"""Tes cepat seluruh komponen memakai dataset sintetis kecil (tanpa GPU)."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from main import run
from src.config import Config
from src.data.dataset import ImageDataset, find_classes, list_images, load_dataset, make_folds
from src.data.preprocess import CropBlackBorder, eval_transform, load_images, train_transform
from src.evaluation.metrics import METRIC_NAMES, compute_metrics, summarize
from src.models import MODEL_NAMES, build_model
from src.training.muon import Muon, newton_schulz
from src.training.trainer import make_optimizers, run_epoch, train_fold
from src.tuning.search import tune

CLASSES = ["glioma", "notumor", "pituitary"]


def make_image(seed: int) -> Image.Image:
    """Gambar acak 48x48 dengan tepi hitam 8 piksel."""
    canvas = np.zeros((48, 48, 3), dtype=np.uint8)
    canvas[8:40, 8:40] = np.random.default_rng(seed).integers(50, 255, (32, 32, 3), dtype=np.uint8)
    return Image.fromarray(canvas)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    for split, n, offset in (("Training", 6, 0), ("Testing", 3, 1000)):
        for c, name in enumerate(CLASSES):
            folder = tmp_path / "data" / split / name
            folder.mkdir(parents=True)
            for i in range(n):
                make_image(offset + c * 100 + i).save(folder / f"{split}_{i}.png")
    return Config(
        data_dir=tmp_path / "data", output_dir=tmp_path / "out", img_size=32, batch_size=4,
        accum_steps=2, epochs=2, patience=1, n_folds=2, pretrained=False, device="cpu", tune_trials=2, tune_epochs=1,
    )


def test_list_images(cfg: Config) -> None:
    classes = find_classes(cfg.data_dir / "Training")
    paths, labels, hashes = list_images(cfg.data_dir / "Training", classes)
    if classes != CLASSES or len(paths) != 18 or len(hashes) != 18 or np.bincount(labels).tolist() != [6, 6, 6]:
        raise AssertionError(f"scan salah: {classes}, {len(paths)}, {np.bincount(labels)}")


def test_duplicates_removed(cfg: Config) -> None:
    train_dir, test_dir = cfg.data_dir / "Training" / "glioma", cfg.data_dir / "Testing" / "glioma"
    shutil.copy(train_dir / "Training_0.png", train_dir / "copy_in_train.png")
    shutil.copy(test_dir / "Testing_0.png", train_dir / "copy_of_test.png")
    _, (paths, _), (test_paths, _) = load_dataset(cfg.data_dir)
    names = {p.name for p in paths if p.parent.name == "glioma"}
    pair_kept = len(names & {"Training_0.png", "copy_in_train.png"})
    if len(paths) != 18 or len(test_paths) != 9 or pair_kept != 1 or "copy_of_test.png" in names:
        raise AssertionError(f"duplikat tidak dibuang: {len(paths)} train, {len(test_paths)} test")


def test_missing_data_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="archive.zip"):
        load_dataset(tmp_path / "kosong")


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
    paths, labels, _ = list_images(cfg.data_dir / "Training", CLASSES)
    images = load_images(paths, 24)
    if len(images) != 18 or {(im.size, im.mode) for im in images} != {((24, 24), "RGB")}:
        raise AssertionError("load_images harus menghasilkan gambar RGB 24x24")
    for transform in (train_transform(), eval_transform()):
        image, label = ImageDataset(images, labels, transform)[0]
        if image.shape != (3, 24, 24) or label != 0:
            raise AssertionError(f"output dataset salah: {image.shape}, {label}")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_build_model(name: str) -> None:
    logits = build_model(name, num_classes=3, img_size=32, pretrained=False, dropout=0.1)(torch.randn(2, 3, 32, 32))
    if logits.shape != (2, 3):
        raise AssertionError(f"shape logits {name} salah: {logits.shape}")


def test_build_model_unknown() -> None:
    with pytest.raises(ValueError, match="tidak dikenal"):
        build_model("resnet25", num_classes=3, img_size=32)


def test_grad_accumulation_equals_big_batch() -> None:
    data = TensorDataset(torch.randn(8, 3, 32, 32), torch.arange(8) % 3)
    weights = []
    for batch_size, accum in ((8, 1), (4, 2)):
        torch.manual_seed(0)
        model = build_model("cnn_scratch", num_classes=3, img_size=32)
        optimizers = [torch.optim.SGD(model.parameters(), lr=0.1)]
        run_epoch(model, DataLoader(data, batch_size), nn.CrossEntropyLoss(), torch.device("cpu"), optimizers, accum)
        weights.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
    if not torch.allclose(weights[0], weights[1], atol=1e-5):
        raise AssertionError("batch 4 x accum 2 harus sama dengan batch 8 x accum 1")


def test_newton_schulz_orthogonal() -> None:
    singular = torch.linalg.svdvals(newton_schulz(torch.randn(64, 288)))
    if singular.min() < 0.5 or singular.max() > 1.5:
        raise AssertionError(f"singular value harus mendekati 1: {singular.min():.3f}..{singular.max():.3f}")


def test_make_optimizers_split(cfg: Config) -> None:
    model = build_model("cnn_modified", num_classes=3, img_size=32)
    muon, adamw = make_optimizers(model, cfg, use_muon=True)
    hidden, rest = muon.param_groups[0]["params"], adamw.param_groups[0]["params"]
    if not isinstance(muon, Muon) or len(hidden) != 4 or any(p.ndim != 4 for p in hidden):
        raise AssertionError("Muon harus memegang tepat 4 filter konvolusi hidden layer")
    if {id(p) for p in hidden} & {id(p) for p in rest} or len(hidden) + len(rest) != len(list(model.parameters())):
        raise AssertionError("setiap parameter harus dipegang tepat satu optimizer")
    if len(make_optimizers(model, cfg, use_muon=False)) != 1:
        raise AssertionError("tanpa Muon cukup satu AdamW")


def test_muon_step_on_conv_filter() -> None:
    conv = nn.Conv2d(8, 16, 3)
    before = conv.weight.detach().clone()
    conv(torch.randn(2, 8, 10, 10)).square().mean().backward()
    Muon([conv.weight], lr=0.02).step()
    if conv.weight.shape != before.shape or not torch.isfinite(conv.weight).all() or torch.equal(conv.weight, before):
        raise AssertionError("Muon harus mengubah filter 4D tanpa mengubah bentuknya")


def test_early_stopping(cfg: Config) -> None:
    loader = DataLoader(TensorDataset(torch.randn(8, 3, 32, 32), torch.arange(8) % 3), 4)
    model = build_model("cnn_scratch", num_classes=3, img_size=32)
    _, history = train_fold(model, loader, loader, replace(cfg, lr=0.0, epochs=20, patience=2), torch.device("cpu"))
    if len(history["val_loss"]) != 3:
        raise AssertionError(f"lr=0 -> val loss tetap -> harus berhenti di epoch 3, bukan {len(history['val_loss'])}")


def test_cuda_required(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA tidak terdeteksi"):
        run(replace(cfg, device="cuda"))


def test_metrics() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    perfect = compute_metrics(y_true, np.eye(3)[y_true])
    if any(abs(perfect[m] - 1.0) > 1e-9 for m in METRIC_NAMES):
        raise AssertionError(f"prediksi sempurna harus bernilai 1: {perfect}")
    summary = summarize([{m: 0.5 for m in METRIC_NAMES}, {m: 1.0 for m in METRIC_NAMES}])
    if summary["f1"] != (0.75, 0.25):
        raise AssertionError(f"rata-rata/std salah: {summary['f1']}")


def test_tune_then_run(cfg: Config) -> None:
    best = tune(replace(cfg, models=("cnn_scratch", "cnn_modified")))
    saved = json.loads((cfg.output_dir / "tuning" / "best_params.json").read_text(encoding="utf-8"))
    scratch_keys = {"lr", "weight_decay", "label_smoothing"}
    modified_keys = scratch_keys | {"muon_lr", "dropout"}
    if saved != best or set(best["cnn_scratch"]) != scratch_keys or set(best["cnn_modified"]) != modified_keys:
        raise AssertionError(f"best_params salah: {saved}")
    tuning = ["optuna.db"] + [f"{m}_{s}" for m in ("cnn_scratch", "cnn_modified") for s in ("trials.md", "history.png")]
    if missing := [n for n in tuning if not (cfg.output_dir / "tuning" / n).exists()]:
        raise AssertionError(f"output tuning tidak lengkap: {missing}")
    if tune(replace(cfg, models=("cnn_scratch",))) != best:
        raise AssertionError("menjalankan ulang harus melanjutkan study lama (sudah 2 trial -> tidak ada trial baru)")

    summaries = run(cfg)
    if set(summaries) != set(MODEL_NAMES) or any(set(s) != set(METRIC_NAMES) for s in summaries.values()):
        raise AssertionError(f"hasil kurang: {summaries}")
    names = ["preprocessing.png", "comparison.png", "results.md"]
    names += [f"figures/{m}/fold{k}_{p}.png" for m in MODEL_NAMES for k in (1, 2) for p in ("history", "cm", "roc")]
    names += [f"models/{m}/fold{k}.pt" for m in MODEL_NAMES for k in (1, 2)]
    if missing := [n for n in names if not (cfg.output_dir / n).exists()]:
        raise AssertionError(f"output tidak lengkap: {missing}")
    if f"{best['cnn_scratch']['lr']:.2e}" not in (cfg.output_dir / "results.md").read_text(encoding="utf-8"):
        raise AssertionError("main.py harus memakai lr hasil tuning")