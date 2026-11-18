"""Tuning hyperparameter dengan Optuna: TPE mencari kombinasi terbaik, MedianPruner menghentikan trial yang jelek.

Hanya fold 1 (train/valid dari folder Training) yang dipakai. Folder Testing tidak pernah disentuh saat tuning,
supaya nilai test di main.py tetap jujur.
"""

import json
from dataclasses import replace

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.data.dataset import ImageDataset, load_dataset, make_folds, make_loader
from src.data.preprocess import eval_transform, load_images, train_transform
from src.models import build_model
from src.training.trainer import get_device, train_fold
from src.tuning.report import save_study_report


def suggest_params(trial: optuna.Trial, name: str, use_muon: bool) -> dict[str, float]:
    """Ruang pencarian: learning rate AdamW (dan Muon), weight decay, label smoothing, dropout (kecuali cnn_scratch)."""
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
    }
    if use_muon:
        params["muon_lr"] = trial.suggest_float("muon_lr", 1e-3, 1e-1, log=True)
    if name != "cnn_scratch":
        params["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
    return params


def tune_model(
    name: str, cfg: Config, n_classes: int, loaders: tuple[DataLoader, DataLoader], device: torch.device
) -> optuna.Study:
    """Cari hyperparameter terbaik satu model berdasarkan akurasi valid pada epoch terbaik.

    Study disimpan di outputs/tuning/optuna.db, jadi jika tuning terhenti, menjalankan ulang akan melanjutkannya.
    """

    use_muon = name in cfg.muon_models

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = replace(cfg, **suggest_params(trial, name, use_muon), epochs=cfg.tune_epochs)
        model = build_model(name, n_classes, cfg.img_size, cfg.pretrained, trial_cfg.dropout)

        def report(epoch: int, val_acc: float) -> None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"akurasi valid di bawah median trial lain pada epoch {epoch}")

        print(f"\n--- {name} | trial {trial.number} | {trial.params}")
        _, history = train_fold(model, *loaders, trial_cfg, device, use_muon, on_epoch=report)
        best = int(np.argmin(history["val_loss"]))
        trial.set_user_attr("best_epoch", best + 1)
        return history["val_acc"][best]

    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{(cfg.output_dir / 'tuning' / 'optuna.db').as_posix()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    done = sum(t.state.is_finished() for t in study.trials)
    study.optimize(objective, n_trials=max(0, cfg.tune_trials - done))
    return study


def tune(cfg: Config) -> dict[str, dict[str, float]]:
    """Tuning semua model di cfg.models, simpan hasil terbaik ke outputs/tuning/best_params.json.

    Returns:
        Hyperparameter terbaik per model.
    """
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)
    out_dir = cfg.output_dir / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    classes, (paths, labels), _ = load_dataset(cfg.data_dir)
    images = load_images(paths, cfg.img_size)
    train_idx, val_idx = make_folds(labels, cfg.n_folds, cfg.seed)[0]
    loaders = (
        make_loader(Subset(ImageDataset(images, labels, train_transform()), train_idx), cfg, train=True),
        make_loader(Subset(ImageDataset(images, labels, eval_transform()), val_idx), cfg),
    )

    best_file = out_dir / "best_params.json"
    best = json.loads(best_file.read_text(encoding="utf-8")) if best_file.exists() else {}
    for name in cfg.models:
        study = tune_model(name, cfg, len(classes), loaders, device)
        best[name] = study.best_params
        save_study_report(study, out_dir)
        print(f"\n{name}: val acc terbaik {study.best_value:.4f} dengan {study.best_params}")
    best_file.write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best