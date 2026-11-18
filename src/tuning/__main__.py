"""Jalankan: python -m src.tuning                       (semua model, 20 trial per model)
          python -m src.tuning --trials 10 --models cnn_scratch
"""

import argparse

from src.config import Config
from src.models import MODEL_NAMES
from src.tuning.search import tune

parser = argparse.ArgumentParser(description="Tuning hyperparameter dengan Optuna + pruning")
parser.add_argument("--trials", type=int, default=Config.tune_trials, help="jumlah trial per model")
parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES, help="model yang di-tuning")
args = parser.parse_args()
tune(Config(tune_trials=args.trials, models=tuple(args.models)))