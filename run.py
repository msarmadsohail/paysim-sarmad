#!/usr/bin/env python
"""
Entry point.

Usage:
  # run all 5 folds — train + generate pools only (default)
  python run.py

  # run a specific fold
  python run.py --fold 0

  # skip ARGN training (use existing weights)
  python run.py --fold 0 --skip-training

  # full pipeline including baseline eval + Optuna HPO
  python run.py --full-pipeline

Live log monitoring:
  tail -f /shared/paysim-sarmad/logs/fold_0.log

MLflow UI:
  mlflow ui --backend-store-uri sqlite:////shared/paysim-sarmad/mlruns/paysim.db --port 5000
"""

import argparse
import json
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/shared/paysim-sarmad/src")

import numpy as np
from config import N_FOLDS, RESULTS_DIR
import tracking as T
from pipeline import run_fold


def aggregate_results() -> None:
    results = []
    for f in range(N_FOLDS):
        p = RESULTS_DIR / f"fold_{f}_results.json"
        if p.exists():
            results.append(json.loads(p.read_text()))

    if not results:
        return

    T.log.info("\n" + "="*60)
    T.log.info("AGGREGATE RESULTS (mean ± std across folds)")
    T.log.info("="*60)

    for clf in ("xgb", "lgbm", "catboost"):
        T.log.info(f"\n── {clf.upper()} ──")
        for split in ("baseline", "augmented"):
            metrics_key = "test"
            vals = {
                m: [r[split][clf][metrics_key][m]
                    for r in results
                    if clf in r.get(split, {})]
                for m in ("pr_auc", "roc_auc", "f1", "f2", "precision", "recall")
            }
            T.log.info(f"  {split}:")
            for m, v in vals.items():
                if v:
                    T.log.info(f"    {m:12s}: {np.mean(v):.4f} ± {np.std(v):.4f}")

    combined_path = RESULTS_DIR / "combined_metrics.json"
    combined_path.write_text(json.dumps(results, indent=2))
    T.log.info(f"\nCombined results → {combined_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="Run a single fold (0-4). Default: all folds.")
    parser.add_argument("--skip-training",   action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--full-pipeline",   action="store_true", help="Also run baseline eval + Optuna HPO after generation.")
    args = parser.parse_args()

    folds = [args.fold] if args.fold is not None else list(range(N_FOLDS))
    stop_after_generation = not args.full_pipeline

    for fold in folds:
        try:
            run_fold(
                fold,
                skip_training=args.skip_training,
                skip_generation=args.skip_generation,
                stop_after_generation=stop_after_generation,
            )
        except Exception as e:
            T.log.error(f"Fold {fold} failed: {e}")
            raise

    if args.fold is None:
        aggregate_results()


if __name__ == "__main__":
    main()
