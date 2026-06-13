"""
Synthetic fraud generation from the three trained ARGN models.

M1 (fraud-only)  : generate freely — all rows are already fraud.
M2 (fraud+10%NF) : generate freely in batches; keep only fraud rows until pool is full.
M3 (full train)  : single free pass, no rebalancing — take whatever fraud rows the
                   model naturally produces. Preserves the model's learned precision/recall.

M1 and M2 target POOL_PER_MODEL fraud rows. M3 is uncapped — natural yield only.
"""

import warnings
from pathlib import Path

import pandas as pd

from config import (
    SYNTH_DIR, TARGET, FRAUD_VAL,
    POOL_PER_MODEL, M2_GEN_BATCH, M3_GEN_BATCH,
    GPU_M1_M2, GPU_M3,
)
import tracking as T


def _filter_fraud(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[TARGET] == FRAUD_VAL].reset_index(drop=True)


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
        T.log.info(f"[fold={fold}] {label} pool: {total}/{target} fraud rows (round {rounds})")
    return pd.concat(collected, ignore_index=True).head(target)


def generate_all(ws_m1: Path, ws_m2: Path, ws_m3: Path, fold: int) -> tuple[Path, Path, Path]:
    warnings.filterwarnings("ignore")
    from train_argn import load_argn

    out_dir = SYNTH_DIR / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    p_m1 = out_dir / "pool_m1.csv"
    p_m2 = out_dir / "pool_m2.csv"
    p_m3 = out_dir / "pool_m3.csv"

    # ── M1: free generation (100% fraud by design) ──────────────────────────
    with T.timed("generate_m1", fold):
        m1 = load_argn(ws_m1, device=f"cuda:{GPU_M1_M2}")
        raw = m1.sample(n_samples=POOL_PER_MODEL)
        pool_m1 = _filter_fraud(raw)
        # top-up if model generated a few non-fraud (shouldn't happen but guard)
        if len(pool_m1) < POOL_PER_MODEL:
            extra = m1.sample(n_samples=(POOL_PER_MODEL - len(pool_m1)) * 2)
            pool_m1 = pd.concat([pool_m1, _filter_fraud(extra)], ignore_index=True).head(POOL_PER_MODEL)
        pool_m1.to_csv(p_m1, index=False)
        T.log.info(f"[fold={fold}] M1 pool saved: {len(pool_m1):,} fraud rows → {p_m1}")

    # ── M2: free batched generation ──────────────────────────────────────────
    with T.timed("generate_m2", fold):
        m2 = load_argn(ws_m2, device=f"cuda:{GPU_M1_M2}")
        pool_m2 = _generate_free_batched(m2, POOL_PER_MODEL, M2_GEN_BATCH, "M2", fold)
        pool_m2.to_csv(p_m2, index=False)
        T.log.info(f"[fold={fold}] M2 pool saved: {len(pool_m2):,} fraud rows → {p_m2}")

    # ── M3: single free pass, no rebalancing — natural yield only ────────────
    with T.timed("generate_m3", fold):
        m3 = load_argn(ws_m3, device=f"cuda:{GPU_M3}")
        raw_m3 = m3.sample(n_samples=M3_GEN_BATCH)
        pool_m3 = _filter_fraud(raw_m3)
        pool_m3.to_csv(p_m3, index=False)
        T.log.info(f"[fold={fold}] M3 pool saved: {len(pool_m3):,} fraud rows → {p_m3}")

    return p_m1, p_m2, p_m3


def load_pools(fold: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir = SYNTH_DIR / f"fold_{fold}"
    return (
        pd.read_csv(out_dir / "pool_m1.csv"),
        pd.read_csv(out_dir / "pool_m2.csv"),
        pd.read_csv(out_dir / "pool_m3.csv"),
    )


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
