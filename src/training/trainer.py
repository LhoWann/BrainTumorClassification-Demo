"""Loop training, validasi, dan prediksi."""

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def autocast(device: torch.device) -> torch.autocast:
    """Mixed precision bfloat16 di GPU (lebih cepat, hemat VRAM); di CPU tidak aktif."""
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    accum_steps: int = 1,
) -> tuple[float, float]:
    """Jalankan satu epoch. Jika optimizer diberikan -> mode train, jika None -> mode evaluasi.

    Gradient accumulation: gradien dari `accum_steps` batch dijumlahkan dulu, baru bobot di-update sekali.
    Hasilnya setara batch besar (batch_size x accum_steps) tetapi hemat memori.

    Returns:
        Rata-rata loss dan akurasi pada epoch ini.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    if training:
        optimizer.zero_grad()
    with torch.set_grad_enabled(training), autocast(device):
        for step, (images, labels) in enumerate(tqdm(loader, desc="train" if training else "valid", leave=False), 1):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                (loss / accum_steps).backward()
                if step % accum_steps == 0 or step == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
    return total_loss / count, correct / count


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    accum_steps: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Latih model, simpan bobot dengan val loss terendah.

    Args:
        model: Model yang akan dilatih.
        train_loader: Data train (sudah teraugmentasi).
        val_loader: Data valid untuk memilih epoch terbaik (tidak dipakai untuk update bobot).
        epochs: Jumlah epoch.
        lr: Learning rate.
        accum_steps: Jumlah batch per update bobot (gradient accumulation).
        device: cpu atau cuda.

    Returns:
        Model dengan bobot terbaik dan riwayat loss/akurasi per epoch.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_loss, best_state = float("inf"), copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, accum_steps)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        for key, value in zip(history, (train_loss, val_loss, train_acc, val_acc), strict=True):
            history[key].append(value)
        print(f"  epoch {epoch:2d} | train loss {train_loss:.4f} acc {train_acc:.4f}", end="")
        print(f" | val loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
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
    with autocast(device):
        for images, y in tqdm(loader, desc="test", leave=False):
            logits = model(images.to(device, non_blocking=True)).float()
            probs.append(torch.softmax(logits, dim=1).cpu())
            labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(probs).numpy()