"""
Optuna HPO for XGBoost, LightGBM, and CatBoost.

Each study optimises:
  - target_pct        : which fraud augmentation rate to use
  - w1, w2, w3        : proportion weights for M1/M2/M3 pool sampling
  - classifier params : model-specific hyperparameters

Objective = PR-AUC on the fixed holdout set.
Baseline (no augmentation) is evaluated once and logged to MLflow before any study.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import TARGET, AUG_TARGETS, OPTUNA_TRIALS, RANDOM_STATE
from data import encode_for_classifier
from evaluate import evaluate_clf
from generate import build_augmented_train

FEATURE_COLS_CACHE: list[str] | None = None


def _feature_cols(df: pd.DataFrame) -> list[str]:
    global FEATURE_COLS_CACHE
    if FEATURE_COLS_CACHE is None:
        FEATURE_COLS_CACHE = [c for c in df.columns if c != TARGET]
    return FEATURE_COLS_CACHE


# ── classifier factories ──────────────────────────────────────────────────────

def _build_xgb(trial: optuna.Trial):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators        = trial.suggest_int("n_estimators", 100, 800),
        max_depth           = trial.suggest_int("max_depth", 3, 10),
        learning_rate       = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        subsample           = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree    = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight    = trial.suggest_int("min_child_weight", 1, 10),
        gamma               = trial.suggest_float("gamma", 0.0, 5.0),
        scale_pos_weight    = trial.suggest_float("scale_pos_weight", 1.0, 50.0),
        reg_alpha           = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda          = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        use_label_encoder   = False,
        eval_metric         = "logloss",
        random_state        = RANDOM_STATE,
        n_jobs              = -1,
    )


def _build_lgbm(trial: optuna.Trial):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators        = trial.suggest_int("n_estimators", 100, 800),
        num_leaves          = trial.suggest_int("num_leaves", 20, 300),
        max_depth           = trial.suggest_int("max_depth", -1, 15),
        learning_rate       = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        min_child_samples   = trial.suggest_int("min_child_samples", 5, 100),
        subsample           = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree    = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha           = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda          = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        is_unbalance        = trial.suggest_categorical("is_unbalance", [True, False]),
        random_state        = RANDOM_STATE,
        n_jobs              = -1,
        verbose             = -1,
    )


def _build_catboost(trial: optuna.Trial):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations          = trial.suggest_int("iterations", 100, 800),
        depth               = trial.suggest_int("depth", 4, 10),
        learning_rate       = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        l2_leaf_reg         = trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0),
        random_strength     = trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        scale_pos_weight    = trial.suggest_float("scale_pos_weight", 1.0, 50.0),
        random_seed         = RANDOM_STATE,
        verbose             = 0,
    )


_BUILDERS = {"xgb": _build_xgb, "lgbm": _build_lgbm, "catboost": _build_catboost}


# ── objective ─────────────────────────────────────────────────────────────────

def _make_objective(
    clf_name: str,
    train_argn: pd.DataFrame,
    holdout: pd.DataFrame,
    pool_m1: pd.DataFrame,
    pool_m2: pd.DataFrame,
    pool_m3: pd.DataFrame,
):
    feat_cols = _feature_cols(train_argn)
    # encode once, reuse across all trials
    train_enc   = encode_for_classifier(train_argn, fit=True)
    holdout_enc = encode_for_classifier(holdout)
    pool_m1_enc = encode_for_classifier(pool_m1)
    pool_m2_enc = encode_for_classifier(pool_m2)
    pool_m3_enc = encode_for_classifier(pool_m3)

    X_val = holdout_enc[feat_cols]
    y_val = holdout_enc[TARGET]

    def objective(trial: optuna.Trial) -> float:
        target_pct = trial.suggest_categorical("target_pct", AUG_TARGETS)
        w1 = trial.suggest_float("w1", 0.0, 1.0)
        w2 = trial.suggest_float("w2", 0.0, 1.0)
        w3 = trial.suggest_float("w3", 0.0, 1.0)

        aug_train = build_augmented_train(
            train_enc, pool_m1_enc, pool_m2_enc, pool_m3_enc,
            w1=w1, w2=w2, w3=w3, target_pct=target_pct,
        )

        clf = _BUILDERS[clf_name](trial)
        clf.fit(aug_train[feat_cols], aug_train[TARGET])

        metrics = evaluate_clf(clf, X_val, y_val)
        return metrics["pr_auc"]

    return objective


# ── public API ────────────────────────────────────────────────────────────────

def run_baseline(
    train_argn: pd.DataFrame,
    holdout: pd.DataFrame,
    test: pd.DataFrame,
    fold: int,
) -> dict:
    """Train each classifier on original data only; log baseline metrics."""
    import mlflow
    feat_cols = _feature_cols(train_argn)
    train_enc   = encode_for_classifier(train_argn, fit=True)
    holdout_enc = encode_for_classifier(holdout)
    test_enc    = encode_for_classifier(test)
    X_train, y_train = train_enc[feat_cols],   train_enc[TARGET]
    X_ho,    y_ho    = holdout_enc[feat_cols],  holdout_enc[TARGET]
    X_test,  y_test  = test_enc[feat_cols],     test_enc[TARGET]

    results = {}
    for name, builder_fn in _BUILDERS.items():
        import optuna
        # use a dummy trial with fixed mid-range params by training a simple default
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier

        clf_map = {
            "xgb":      XGBClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, use_label_encoder=False, eval_metric="logloss"),
            "lgbm":     LGBMClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            "catboost": CatBoostClassifier(iterations=300, random_seed=RANDOM_STATE, verbose=0),
        }
        clf = clf_map[name]
        clf.fit(X_train, y_train)
        ho_m  = evaluate_clf(clf, X_ho, y_ho)
        tst_m = evaluate_clf(clf, X_test, y_test)

        with mlflow.start_run(run_name=f"baseline_{name}_fold{fold}", nested=True):
            mlflow.log_params({"classifier": name, "fold": fold, "augmented": False})
            for k, v in tst_m.items():
                mlflow.log_metric(f"test_{k}", v)
            for k, v in ho_m.items():
                mlflow.log_metric(f"holdout_{k}", v)

        results[name] = {"holdout": ho_m, "test": tst_m}

    return results


def run_studies(
    train_argn: pd.DataFrame,
    holdout: pd.DataFrame,
    test: pd.DataFrame,
    pool_m1: pd.DataFrame,
    pool_m2: pd.DataFrame,
    pool_m3: pd.DataFrame,
    fold: int,
) -> dict:
    """Run one Optuna study per classifier; return best configs and test metrics."""
    import mlflow
    feat_cols = _feature_cols(train_argn)
    test_enc  = encode_for_classifier(test, fit=False)
    X_test, y_test = test_enc[feat_cols], test_enc[TARGET]

    best_results = {}

    for clf_name in ("xgb", "lgbm", "catboost"):
        study = optuna.create_study(
            direction="maximize",
            study_name=f"fold{fold}_{clf_name}",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        obj = _make_objective(clf_name, train_argn, holdout, pool_m1, pool_m2, pool_m3)

        with mlflow.start_run(run_name=f"optuna_{clf_name}_fold{fold}", nested=True):
            mlflow.log_params({"classifier": clf_name, "fold": fold, "n_trials": OPTUNA_TRIALS})
            study.optimize(obj, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

            best = study.best_trial
            mlflow.log_metric("best_holdout_pr_auc", best.value)
            mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})

            # retrain best config on full train (train_argn + holdout) + optimal synthetic
            full_train = pd.concat([train_argn, holdout], ignore_index=True)
            full_train_enc = encode_for_classifier(full_train, fit=False)
            pool_m1_enc = encode_for_classifier(pool_m1, fit=False)
            pool_m2_enc = encode_for_classifier(pool_m2, fit=False)
            pool_m3_enc = encode_for_classifier(pool_m3, fit=False)
            aug_full = build_augmented_train(
                full_train_enc, pool_m1_enc, pool_m2_enc, pool_m3_enc,
                w1=best.params["w1"],
                w2=best.params["w2"],
                w3=best.params["w3"],
                target_pct=best.params["target_pct"],
            )

            # rebuild clf with best params (re-suggest from frozen trial)
            final_trial = optuna.trial.FixedTrial(best.params)
            clf_final = _BUILDERS[clf_name](final_trial)
            clf_final.fit(aug_full[feat_cols], aug_full[TARGET])

            test_m = evaluate_clf(clf_final, X_test, y_test)
            for k, v in test_m.items():
                mlflow.log_metric(f"test_{k}", v)

        best_results[clf_name] = {
            "best_params": best.params,
            "holdout_pr_auc": best.value,
            "test": test_m,
        }

    return best_results
