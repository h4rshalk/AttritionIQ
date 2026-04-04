import pandas as pd
import numpy as np
import pickle
import os
import sys
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import xgboost as xgb
import lightgbm as lgb
import optuna

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from feature_pipeline import run_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("attrition-risk-engine")

def get_data():
    X, y, scaler, feature_cols = run_pipeline()
    X = pd.DataFrame(X, columns=feature_cols)
    return train_test_split(X, y, test_size=0.2,
                            random_state=42, stratify=y)

def tune_xgb(X_train, y_train):
    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        model = xgb.XGBClassifier(**params)
        return cross_val_score(model, X_train, y_train,
                               cv=3, scoring='roc_auc').mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    return study.best_params

def train_and_log(model, model_name, params,
                  X_train, X_test, y_train, y_test, feature_cols):

    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, proba)
        f1    = f1_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        # Log params
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics({
            "roc_auc":   round(auc, 4),
            "f1_score":  round(f1, 4),
            "precision": round(report['1']['precision'], 4),
            "recall":    round(report['1']['recall'], 4),
        })

        # Log model
        if model_name.startswith("XGB"):
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.lightgbm.log_model(model, artifact_path="model")

        # Save locally too
        os.makedirs("artifacts", exist_ok=True)
        model_path = f"artifacts/{model_name.lower().replace(' ','_')}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"{model_name} — AUC: {auc:.4f} | F1: {f1:.4f}")
        return auc, model

def main():
    X_train, X_test, y_train, y_test = get_data()
    feature_cols = list(X_train.columns)

    # --- Run 1: Baseline XGBoost ---
    xgb_base_params = {
        'n_estimators': 200, 'max_depth': 6,
        'learning_rate': 0.05, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'use_label_encoder': False,
        'eval_metric': 'logloss', 'random_state': 42
    }
    xgb_base = xgb.XGBClassifier(**xgb_base_params)
    train_and_log(xgb_base, "XGB Baseline", xgb_base_params,
                  X_train, X_test, y_train, y_test, feature_cols)

    # --- Run 2: Baseline LightGBM ---
    lgb_params = {
        'n_estimators': 200, 'max_depth': 6,
        'learning_rate': 0.05, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'random_state': 42, 'verbose': -1
    }
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    train_and_log(lgb_model, "LightGBM Baseline", lgb_params,
                  X_train, X_test, y_train, y_test, feature_cols)

    # --- Run 3: Tuned XGBoost (best model) ---
    print("\nRunning Optuna tuning (30 trials)...")
    best_params = tune_xgb(X_train, y_train)
    best_params.update({'use_label_encoder': False,
                        'eval_metric': 'logloss', 'random_state': 42})
    xgb_tuned = xgb.XGBClassifier(**best_params)
    best_auc, best_model = train_and_log(
        xgb_tuned, "XGB Tuned (Optuna)", best_params,
        X_train, X_test, y_train, y_test, feature_cols)

    print(f"\nBest model saved. Final AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()