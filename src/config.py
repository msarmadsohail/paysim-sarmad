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
POOL_PER_MODEL  = 20_000      # target fraud rows to collect per model
M2_GEN_BATCH    = 50_000      # rows generated per batch while filling M2 pool
M3_REBAL_PROB   = 0.5         # isFraud probability for M3 RebalancingConfig

# GPU assignment
GPU_M1_M2 = 0
GPU_M3    = 1

# Optuna
OPTUNA_TRIALS = 100
AUG_TARGETS   = [0.0020, 0.0030, 0.0050, 0.0100]   # 0.20 / 0.30 / 0.50 / 1.00 %

# metrics
F_BETA = 2
