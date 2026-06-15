"""
Synthetic fraud generation from the three trained ARGN models.

M1 (fraud-only)  : batched free generation — target M1_POOL_TARGET fraud rows, GPU 0.
M2 (fraud+10%NF) : batched free generation — target M2_POOL_TARGET fraud rows, GPU 1.
M3 (full train)  : single free pass of M3_GEN_BATCH samples, natural yield, GPU 2.
                   No rebalancing — preserves learned precision/recall.

All three models generate in parallel (one thread each). Folds run sequentially.
"""

import warnings
import threading
from pathlib import Path

import pandas as pd

from config import (
    SYNTH_DIR, TARGET, FRAUD_VAL,
    M1_POOL_TARGET, M2_POOL_TARGET,
    M1_GEN_BATCH, M2_GEN_BATCH, M3_GEN_BATCH,
    POOL_PER_MODEL,
    GPU_M1, GPU_M2, GPU_M3,
)
import tracking as T


def _filter_fraud(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[TARGET] == FRAUD_VAL].reset_index(drop=True)


def _recompute_newbalanceOrig(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute newbalanceOrig post-generation using type-aware formula.
    CASH_IN adds to sender balance; all others drain it (floored at 0).
    """
    df = df.copy()
    cash_in_mask = df["type"].str.upper() == "CASH_IN"
    df["newbalanceOrig"] = 0.0
    df.loc[cash_in_mask, "newbalanceOrig"] = df.loc[cash_in_mask, "oldbalanceOrg"] + df.loc[cash_in_mask, "amount"]
    df.loc[~cash_in_mask, "newbalanceOrig"] = (df.loc[~cash_in_mask, "oldbalanceOrg"] - df.loc[~cash_in_mask, "amount"]).clip(lower=0)
    return df


def _generate_free_batched(argn, target: int, batch: int, label: str, fold: int) -> pd.DataFrame:
    """Generate in batches until `target` fraud rows are collected."""
    collected: list[pd.DataFrame] = []
    total = 0
    rounds = 0
    while total < target:
        chunk = argn.sample(n_samples=batch)
        fraud_chunk = _filter_fraud(chunk)
        collected.append(fraud_chunk)
        total += len(fraud_chunk)
        rounds += 1
        T.log.info(f"[fold={fold}] {label} pool: {total:,}/{target:,} fraud rows (round {rounds})")
    return pd.concat(collected, ignore_index=True).head(target)


def _run_m1(ws_m1: Path, fold: int, out_path: Path, results: dict) -> None:
    warnings.filterwarnings("ignore")
    from train_argn import load_argn
    with T.timed("generate_m1", fold):
        m1 = load_argn(ws_m1, device=f"cuda:{GPU_M1}")
        pool = _generate_free_batched(m1, M1_POOL_TARGET, M1_GEN_BATCH, "M1", fold)
        pool.to_csv(out_path, index=False)
        T.log.info(f"[fold={fold}] M1 pool saved: {len(pool):,} fraud rows → {out_path}")
        results["m1"] = pool


def _run_m2(ws_m2: Path, fold: int, out_path: Path, results: dict) -> None:
    warnings.filterwarnings("ignore")
    from train_argn import load_argn
    with T.timed("generate_m2", fold):
        m2 = load_argn(ws_m2, device=f"cuda:{GPU_M2}")
        pool = _generate_free_batched(m2, M2_POOL_TARGET, M2_GEN_BATCH, "M2", fold)
        pool = _recompute_newbalanceOrig(pool)
        pool.to_csv(out_path, index=False)
        T.log.info(f"[fold={fold}] M2 pool saved: {len(pool):,} fraud rows → {out_path}")
        results["m2"] = pool


def _run_m3(ws_m3: Path, fold: int, out_path: Path, results: dict) -> None:
    warnings.filterwarnings("ignore")
    from train_argn import load_argn
    with T.timed("generate_m3", fold):
        m3 = load_argn(ws_m3, device=f"cuda:{GPU_M3}")
        raw = m3.sample(n_samples=M3_GEN_BATCH)
        pool = _filter_fraud(raw)
        pool = _recompute_newbalanceOrig(pool)
        pool.to_csv(out_path, index=False)
        T.log.info(f"[fold={fold}] M3 pool saved: {len(pool):,} fraud rows (natural yield from {M3_GEN_BATCH:,} samples) → {out_path}")
        results["m3"] = pool


def generate_all(ws_m1: Path, ws_m2: Path, ws_m3: Path, fold: int) -> tuple[Path, Path, Path]:
    out_dir = SYNTH_DIR / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    p_m1 = out_dir / "pool_m1.csv"
    p_m2 = out_dir / "pool_m2.csv"
    p_m3 = out_dir / "pool_m3.csv"

    results = {}

    # M1 only — single GPU, no threading needed
    T.log.info(f"[fold={fold}] Generating M1 only on cuda:{GPU_M1}")
    _run_m1(ws_m1, fold, p_m1, results)

    return p_m1, p_m2, p_m3


def load_pools(fold: int) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    out_dir = SYNTH_DIR / f"fold_{fold}"
    def _load(name):
        p = out_dir / name
        return pd.read_csv(p) if p.exists() else None
    return _load("pool_m1.csv"), _load("pool_m2.csv"), _load("pool_m3.csv")


def build_augmented_train(
    train_orig: pd.DataFrame,
    pool_m1: pd.DataFrame,
    pool_m2: pd.DataFrame,
    pool_m3: pd.DataFrame,
    w1: float,
    w2: float,
    w3: float,
    target_pct: float,
) -> pd.DataFrame:
    """
    Select rows from each pool proportional to (w1, w2, w3) so total fraud reaches target_pct.
    Returns original train + selected synthetic fraud rows (isFraud=1 rows only from pools).
    """
    n_orig_fraud = int((train_orig[TARGET] == FRAUD_VAL).sum())
    n_total      = len(train_orig)
    n_target_fraud = int(n_total * target_pct)
    n_synthetic  = max(0, n_target_fraud - n_orig_fraud)

    if n_synthetic == 0:
        return train_orig.copy()

    total_w = w1 + w2 + w3 + 1e-9
    n1 = min(int(n_synthetic * w1 / total_w), len(pool_m1))
    n2 = min(int(n_synthetic * w2 / total_w), len(pool_m2))
    n3 = min(int(n_synthetic * w3 / total_w), len(pool_m3))

    parts = [train_orig]
    if n1 > 0:
        parts.append(pool_m1.sample(n=n1, random_state=42))
    if n2 > 0:
        parts.append(pool_m2.sample(n=n2, random_state=42))
    if n3 > 0:
        parts.append(pool_m3.sample(n=n3, random_state=42))

    return pd.concat(parts, ignore_index=True)
