#!/usr/bin/env python
"""
Parallel fold runner — uses all 4 GPUs by running 2 folds simultaneously.

GPU layout per fold pair:
  Fold A  →  training: M1+M2 on GPU 0, M3 on GPU 1
             generation: M1 on GPU 0, M2 on GPU 1, M3 on GPU 1 (sequential)
  Fold B  →  training: M1+M2 on GPU 2, M3 on GPU 3
             generation: M1 on GPU 2, M2 on GPU 3, M3 on GPU 3 (sequential)

Schedule (train → generate before starting next pair):
  Round 1: folds 0 + 1  (GPUs 0,1 | 2,3)
  Round 2: folds 2 + 3  (GPUs 0,1 | 2,3)
  Round 3: fold  4      (GPUs 0,1)

Generation is free — M1/M2 batched until pool target, M3 single natural pass.
"""

import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, "/shared/paysim-sarmad/src")


def _run_fold_worker(fold: int, gpu_primary: int, gpu_secondary: int) -> None:
    """Worker that runs in a subprocess with overridden GPU assignments."""
    # M1-only run: each fold gets one dedicated physical GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_primary}"
    os.environ["ARGN_GPU_M1"] = "0"
    os.environ["ARGN_GPU_M2"] = "0"
    os.environ["ARGN_GPU_M3"] = "0"

    import warnings
    warnings.filterwarnings("ignore")

    import tracking as T
    T.setup_fold_logging(fold)
    T.log.info(f"[fold={fold}] process started — physical GPU {gpu_primary} → cuda:0")

    from pipeline import run_fold
    run_fold(fold, skip_training=False, skip_generation=False, stop_after_generation=True)


def _run_pair(fold_a: int, fold_b: int | None) -> None:
    """Launch fold_a on GPUs 0,1 and fold_b (if given) on GPUs 2,3 — wait for both."""
    ctx = mp.get_context("spawn")
    procs = []

    # M1-only: fold_a on GPU 0, fold_b on GPU 1 — each gets its own dedicated GPU
    p_a = ctx.Process(
        target=_run_fold_worker,
        args=(fold_a, 0, 0),
        name=f"fold-{fold_a}",
    )
    p_a.start()
    procs.append((fold_a, p_a))

    if fold_b is not None:
        p_b = ctx.Process(
            target=_run_fold_worker,
            args=(fold_b, 1, 1),
            name=f"fold-{fold_b}",
        )
        p_b.start()
        procs.append((fold_b, p_b))

    for fold, p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Fold {fold} worker exited with code {p.exitcode}")
        print(f"[run_parallel] fold {fold} complete", flush=True)


def main() -> None:
    rounds = [
        (2, 3),
        (4, None),
    ]

    total_start = time.time()
    for fold_a, fold_b in rounds:
        label = f"folds {fold_a}+{fold_b}" if fold_b is not None else f"fold {fold_a}"
        print(f"\n{'='*60}", flush=True)
        print(f"[run_parallel] Starting {label}", flush=True)
        print(f"{'='*60}", flush=True)
        _run_pair(fold_a, fold_b)
        print(f"[run_parallel] {label} done", flush=True)

    elapsed = time.time() - total_start
    print(f"\n[run_parallel] All 5 folds complete in {elapsed/3600:.2f}h", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
