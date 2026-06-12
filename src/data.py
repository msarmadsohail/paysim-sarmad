import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import (
    DATA_DIR, DROP_COLS, TARGET, FRAUD_VAL, STRAT_COL,
    HOLDOUT_RATIO, RANDOM_STATE, M2_NONFR_FRAC,
)

_TYPE_ENCODER: LabelEncoder | None = None
_CAT_COLS = ["type"]


def load_fold(fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "splits" / str(fold) / "train.csv")
    test  = pd.read_csv(DATA_DIR / "splits" / str(fold) / "test.csv")
    return train, test


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=DROP_COLS, errors="ignore").reset_index(drop=True)


def split_holdout(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_class = train[TARGET].value_counts().min()
    stratify = train[TARGET] if min_class >= 2 else None
    train_argn, holdout = train_test_split(
        train,
        test_size=HOLDOUT_RATIO,
        stratify=stratify,
        random_state=RANDOM_STATE,
    )
    return train_argn.reset_index(drop=True), holdout.reset_index(drop=True)


def build_m1(train: pd.DataFrame) -> pd.DataFrame:
    """All fraud rows only."""
    return train[train[TARGET] == FRAUD_VAL].reset_index(drop=True)


def build_m2(train: pd.DataFrame) -> pd.DataFrame:
    """All fraud + 10% non-fraud stratified by transaction type."""
    fraud    = train[train[TARGET] == FRAUD_VAL]
    non_fraud = train[train[TARGET] != FRAUD_VAL]

    sample = (
        non_fraud
        .groupby(STRAT_COL, group_keys=False)
        .apply(lambda g: g.sample(frac=M2_NONFR_FRAC, random_state=RANDOM_STATE))
    ).reset_index(drop=True)

    return pd.concat([fraud, sample], ignore_index=True)


def build_m3(train: pd.DataFrame) -> pd.DataFrame:
    """Full training split — no modification."""
    return train.reset_index(drop=True)


def encode_for_classifier(df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """Label-encode categorical columns so XGB/LGBM/CatBoost accept the data."""
    global _TYPE_ENCODER
    df = df.copy()
    for col in _CAT_COLS:
        if col not in df.columns:
            continue
        if fit or _TYPE_ENCODER is None:
            _TYPE_ENCODER = LabelEncoder().fit(df[col].astype(str))
        df[col] = _TYPE_ENCODER.transform(df[col].astype(str))
    return df


def fraud_stats(df: pd.DataFrame, label: str) -> str:
    n_fraud = (df[TARGET] == FRAUD_VAL).sum()
    return f"{label}: {len(df):,} rows | fraud={n_fraud:,} ({n_fraud/len(df)*100:.3f}%)"
