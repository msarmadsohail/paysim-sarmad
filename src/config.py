from pathlib import Path

BASE_DIR   = Path("/shared/paysim-sarmad")
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SYNTH_DIR  = BASE_DIR / "synthetic"
RESULTS_DIR= BASE_DIR / "results"
MLRUNS_DIR = BASE_DIR / "mlruns"
LOG_DIR    = BASE_DIR / "logs"

N_FOLDS = 5

# columns
DROP_COLS  = ["nameOrig", "nameDest"]
TARGET     = "isFraud"
FRAUD_VAL  = 1
STRAT_COL  = "type"          # used for M2 stratified non-fraud sampling

# holdout split inside each fold's train.csv
HOLDOUT_RATIO = 0.20
RANDOM_STATE  = 42

# ARGN training
M1_MAX_EPOCHS = 50            # fraud-only; built-in early stop (patience=4) fires first
M2_MAX_EPOCHS = 100
M3_MAX_EPOCHS = 100
M2_NONFR_FRAC = 0.10          # 10% of non-fraud rows for M2

# generation
M1_POOL_TARGET  = 150_000     # target fraud rows — M1 batched free generation
M2_POOL_TARGET  = 150_000     # target fraud rows — M2 batched free generation
M1_GEN_BATCH    = 150_000     # batch size for M1 (100% fraud, one pass usually enough)
M2_GEN_BATCH    = 50_000      # batch size for M2 (~1.3% fraud rate in output)
M3_GEN_BATCH    = 5_000_000   # single free pass for M3 — natural yield, no target cap

# legacy alias kept for build_augmented_train compatibility
POOL_PER_MODEL  = 150_000

# GPU assignment — all 3 models run in parallel
GPU_M1 = 0
GPU_M2 = 1
GPU_M3 = 2

# Optuna
OPTUNA_TRIALS = 100
AUG_TARGETS   = [0.0020, 0.0030, 0.0050, 0.0100]   # 0.20 / 0.30 / 0.50 / 1.00 %

# metrics
F_BETA = 2
