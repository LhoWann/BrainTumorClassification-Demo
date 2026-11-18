"""Loop training (early stopping + penurunan learning rate), validasi, dan prediksi."""

import copy
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.training.muon import Muon


def make_optimizers(model: nn.Module, cfg: Config, use_muon: bool) -> list[torch.optim.Optimizer]:
    """AdamW untuk semua bobot, atau (use_muon) Muon untuk filter konvolusi hidden layer + AdamW untuk sisanya.

    Hidden layer = bobot >= 2 dimensi di `model.features`. Bias, BatchNorm, dan layer output (classifier) tetap AdamW.
    """
    adamw = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    if not use_muon:
        return [torch.optim.AdamW(model.parameters(), **adamw)]
    params = list(model.named_parameters())
    hidden = [p for n, p in params if n.startswith("features.") and p.ndim >= 2]
    others = [p for n, p in params if not (n.startswith("features.") and p.ndim >= 2)]
    return [Muon(hidden, lr=cfg.muon_lr, weight_decay=cfg.weight_decay), torch.optim.AdamW(others, **adamw)]


def get_device(name: str) -> torch.device:
    """Pilih device. Untuk "cuda", berhenti dengan pesan jelas jika GPU tidak terdeteksi."""
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA tidak terdeteksi: install torch CUDA (lihat README) atau set device='cpu'")
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return torch.device(name)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizers: list[torch.optim.Optimizer] | None = None,
    accum_steps: int = 1,
) -> tuple[float, float]:
    """Jalankan satu epoch. Jika optimizers diberikan -> mode train, jika None -> mode evaluasi.

    Gradient accumulation: gradien dari `accum_steps` batch dijumlahkan dulu, baru bobot di-update sekali.
    Hasilnya setara batch besar (batch_size x accum_steps) tetapi hemat memori.

    Returns:
        Rata-rata loss dan akurasi pada epoch ini.
    """
    training = optimizers is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    if training:
        model.zero_grad()
    with torch.set_grad_enabled(training):
        for step, (images, labels) in enumerate(tqdm(loader, desc="train" if training else "valid", leave=False), 1):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                (loss / accum_steps).backward()
                if step % accum_steps == 0 or step == len(loader):
                    for optimizer in optimizers:
                        optimizer.step()
                    model.zero_grad()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
    return total_loss / count, correct / count


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    use_muon: bool = False,
    on_epoch: Callable[[int, float], None] | None = None,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Latih model sampai val loss berhenti membaik (early stopping), lalu kembalikan bobot terbaiknya.

    Args:
        model: Model yang akan dilatih.
        train_loader: Data train (sudah teraugmentasi).
        val_loader: Data valid untuk memilih epoch terbaik (tidak dipakai untuk update bobot).
        cfg: Memakai lr, muon_lr, weight_decay, label_smoothing, epochs, patience, dan accum_steps.
        device: cpu atau cuda.
        use_muon: Muon untuk filter konvolusi hidden layer + AdamW untuk sisanya (lihat make_optimizers).
        on_epoch: Dipanggil tiap epoch dengan (epoch, val_acc); dipakai Optuna untuk pruning.

    Returns:
        Model dengan bobot terbaik dan riwayat loss/akurasi/learning rate AdamW per epoch.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizers = make_optimizers(model, cfg, use_muon)
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(o, factor=0.5, patience=2) for o in optimizers]
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_loss, best_state, wait = float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, cfg.epochs + 1):
        lrs = [o.param_groups[0]["lr"] for o in optimizers]
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizers, cfg.accum_steps)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        for scheduler in schedulers:
            scheduler.step(val_loss)
        for key, value in zip(history, (train_loss, val_loss, train_acc, val_acc, lrs[-1]), strict=True):
            history[key].append(value)
        lr_text = "/".join(f"{lr:.1e}" for lr in lrs)
        print(f"  epoch {epoch:2d} | lr {lr_text} | train loss {train_loss:.4f} acc {train_acc:.4f}", end="")
        print(f" | val loss {val_loss:.4f} acc {val_acc:.4f}")
        if on_epoch is not None:
            on_epoch(epoch, val_acc)
        if val_loss < best_loss:
            best_loss, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"  early stopping: val loss tidak membaik {cfg.patience} epoch")
                break
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Prediksi probabilitas setiap kelas.

    Returns:
        Label asli (N,) dan probabilitas softmax (N, jumlah_kelas).
    """
    model.eval()
    labels, probs = [], []
    for images, y in tqdm(loader, desc="test", leave=False):
        probs.append(torch.softmax(model(images.to(device, non_blocking=True)), dim=1).cpu())
        labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(probs).numpy()