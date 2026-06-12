"""
Per-fold orchestrator.

Flow
----
1. Load & preprocess fold data
2. Split holdout (20% stratified)
3. Build M1/M2/M3 datasets
4. Train all 3 ARGNs  (M3 parallel on GPU 1)
5. Generate fraud pools from each model
6. Run baseline classifiers (original data only) → log to MLflow
7. Run Optuna studies (3 classifiers × 100 trials) → log to MLflow
8. Save per-fold results JSON
"""

import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import mlflow
import pandas as pd

from config import RESULTS_DIR, LOG_DIR
import tracking as T
from data import (
    load_fold, preprocess, split_holdout,
    build_m1, build_m2, build_m3, fraud_stats,
)
from train_argn import train_all
from generate import generate_all, load_pools
from optimize import run_baseline, run_studies


def run_fold(fold: int, skip_training: bool = False, skip_generation: bool = False) -> dict:
    T.setup_fold_logging(fold)
    T.init_mlflow(fold)

    fold_start = time.time()
    T.log.info(f"{'='*60}")
    T.log.info(f"FOLD {fold} — pipeline start")
    T.log.info(f"{'='*60}")

    with mlflow.start_run(run_name=f"fold_{fold}"):
        mlflow.log_param("fold", fold)

        # ── 1. Data ───────────────────────────────────────────────────────────
        with T.timed("load_data", fold):
            train_raw, test_raw = load_fold(fold)
            train = preprocess(train_raw)
            test  = preprocess(test_raw)
            T.log.info(fraud_stats(train, "train"))
            T.log.info(fraud_stats(test,  "test"))
            mlflow.log_params({
                "train_rows": len(train),
                "test_rows":  len(test),
                "train_fraud": int((train["isFraud"] == 1).sum()),
            })

        # ── 2. Holdout split ─────────────────────────────────────────────────
        train_argn, holdout = split_holdout(train)
        T.log.info(fraud_stats(train_argn, "train_argn"))
        T.log.info(fraud_stats(holdout, "holdout"))

        # ── 3. ARGN datasets ─────────────────────────────────────────────────
        m1_data = build_m1(train_argn)
        m2_data = build_m2(train_argn)
        m3_data = build_m3(train_argn)
        T.log.info(fraud_stats(m1_data, "M1"))
        T.log.info(fraud_stats(m2_data, "M2"))
        T.log.info(fraud_stats(m3_data, "M3"))
        mlflow.log_params({
            "m1_rows": len(m1_data),
            "m2_rows": len(m2_data),
            "m3_rows": len(m3_data),
        })

        # ── 4. Training ───────────────────────────────────────────────────────
        from config import MODELS_DIR
        ws_m1 = MODELS_DIR / f"fold_{fold}" / "m1"
        ws_m2 = MODELS_DIR / f"fold_{fold}" / "m2"
        ws_m3 = MODELS_DIR / f"fold_{fold}" / "m3"

        if not skip_training:
            with T.timed("train_all_argns", fold):
                ws_m1, ws_m2, ws_m3 = train_all(m1_data, m2_data, m3_data, fold)
        else:
            T.log.info(f"[fold={fold}] skip_training=True — using existing weights")

        # ── 5. Generation ─────────────────────────────────────────────────────
        if not skip_generation:
            with T.timed("generate_all_pools", fold):
                generate_all(ws_m1, ws_m2, ws_m3, fold)
        else:
            T.log.info(f"[fold={fold}] skip_generation=True — using existing pools")

        pool_m1, pool_m2, pool_m3 = load_pools(fold)
        mlflow.log_params({
            "pool_m1_size": len(pool_m1),
            "pool_m2_size": len(pool_m2),
            "pool_m3_size": len(pool_m3),
        })

        # ── 6. Baseline ───────────────────────────────────────────────────────
        with T.timed("baseline_eval", fold):
            baseline = run_baseline(train_argn, holdout, test, fold)
        T.log.info(f"[fold={fold}] Baseline results:")
        for clf, res in baseline.items():
            T.log.info(f"  {clf}: test pr_auc={res['test']['pr_auc']:.4f}  roc_auc={res['test']['roc_auc']:.4f}")

        # ── 7. Optuna studies ─────────────────────────────────────────────────
        with T.timed("optuna_studies", fold):
            augmented = run_studies(
                train_argn, holdout, test,
                pool_m1, pool_m2, pool_m3,
                fold,
            )
        T.log.info(f"[fold={fold}] Augmented results:")
        for clf, res in augmented.items():
            T.log.info(f"  {clf}: test pr_auc={res['test']['pr_auc']:.4f}  best_pct={res['best_params']['target_pct']}")

        # ── 8. Save results ───────────────────────────────────────────────────
        total_elapsed = round(time.time() - fold_start, 1)
        T.log.info(f"[fold={fold}] TOTAL elapsed: {total_elapsed}s  ({total_elapsed/3600:.2f}h)")
        mlflow.log_metric("total_elapsed_s", total_elapsed)

        fold_result = {
            "fold": fold,
            "total_elapsed_s": total_elapsed,
            "baseline":  baseline,
            "augmented": augmented,
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"fold_{fold}_results.json"
        out_path.write_text(json.dumps(fold_result, indent=2))
        T.log.info(f"[fold={fold}] Results saved → {out_path}")

        mlflow.log_artifact(str(out_path))

    return fold_result
