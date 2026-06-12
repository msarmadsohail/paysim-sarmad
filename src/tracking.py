import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import mlflow

from config import LOG_DIR, MLRUNS_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("paysim")


def _jsonl(fold: int) -> Path:
    return LOG_DIR / f"fold_{fold}.jsonl"


def _append(fold: int, entry: dict) -> None:
    with open(_jsonl(fold), "a") as f:
        f.write(json.dumps(entry) + "\n")


def setup_fold_logging(fold: int) -> None:
    """Add a per-fold file handler so logs are streamable via `tail -f`."""
    fh = logging.FileHandler(LOG_DIR / f"fold_{fold}.log", mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger("paysim").addHandler(fh)
    # also capture mostlyai logs
    logging.getLogger("mostlyai").addHandler(fh)


def init_mlflow(fold: int) -> None:
    # SQLite backend — supported in all MLflow versions, shareable across users
    db_path = MLRUNS_DIR / "paysim.db"
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(f"paysim-fold-{fold}")


@contextmanager
def timed(name: str, fold: int, extra: dict | None = None):
    t0 = time.time()
    log.info(f"[fold={fold}] ▶ START {name}")
    _append(fold, {"stage": name, "fold": fold, "status": "started", "ts": t0, **(extra or {})})
    try:
        yield
        elapsed = round(time.time() - t0, 1)
        log.info(f"[fold={fold}] ✔ DONE  {name}  ({elapsed}s)")
        _append(fold, {"stage": name, "fold": fold, "status": "done", "elapsed_s": elapsed, "ts": time.time()})
        if mlflow.active_run():
            mlflow.log_metric(f"{name}_elapsed_s", elapsed)
    except Exception as exc:
        elapsed = round(time.time() - t0, 1)
        log.error(f"[fold={fold}] ✖ FAIL  {name}  ({elapsed}s): {exc}")
        _append(fold, {"stage": name, "fold": fold, "status": "failed", "elapsed_s": elapsed, "error": str(exc), "ts": time.time()})
        raise
