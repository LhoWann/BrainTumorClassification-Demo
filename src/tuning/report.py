"""Simpan ringkasan tuning: tabel semua trial (Markdown) dan grafik riwayat optimasi + pentingnya parameter."""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

PARAMS = ("lr", "muon_lr", "weight_decay", "label_smoothing", "dropout")


def save_study_report(study: optuna.Study, out_dir: Path) -> None:
    """Tulis <model>_trials.md, <model>_history.png, dan <model>_importance.png ke out_dir."""
    rows = ["| trial | status | val acc | epoch | " + " | ".join(PARAMS) + " |", "|---" * (len(PARAMS) + 4) + "|"]
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            status, acc, epoch = "selesai", t.value, t.user_attrs["best_epoch"]
        elif t.state == optuna.trial.TrialState.PRUNED and t.intermediate_values:
            epoch = max(t.intermediate_values)
            status, acc = "dipangkas", t.intermediate_values[epoch]
        else:
            continue
        cells = [f"{t.params[p]:.2e}" if p in t.params else "-" for p in PARAMS]
        rows.append(f"| {t.number} | {status} | {acc:.4f} | {epoch} | " + " | ".join(cells) + " |")
    (out_dir / f"{study.study_name}_trials.md").write_text("\n".join(rows), encoding="utf-8")

    plots = [("history", plot_optimization_history)]
    if len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))) > 1:
        plots.append(("importance", plot_param_importances))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        for suffix, plot in plots:
            ax = plot(study)
            ax.set_title(f"{study.study_name} - {suffix}")
            ax.figure.tight_layout()
            ax.figure.savefig(out_dir / f"{study.study_name}_{suffix}.png", dpi=120)
            plt.close(ax.figure)