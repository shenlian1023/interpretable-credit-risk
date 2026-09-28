from pathlib import Path
from pprint import pformat
import time
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from preprocessing import prepare_data


RANDOM_STATE = 42
OUTPUT_TABLES_DIR = Path("outputs/tables/experiment_1")
SCORING = "average_precision"
CV_SPLITS = 3
GRID_SEARCH_N_JOBS = 1
MODEL_N_JOBS = 2


def get_optional_model_classes():
    """Import optional models only if they are installed."""
    optional_models = {}

    try:
        from xgboost import XGBClassifier

        optional_models["XGBoost"] = XGBClassifier
    except ImportError:
        print("XGBoost is not installed. Skipping XGBoost.")

    try:
        from lightgbm import LGBMClassifier

        optional_models["LightGBM"] = LGBMClassifier
    except ImportError:
        print("LightGBM is not installed. Skipping LightGBM.")

    try:
        from interpret.glassbox import ExplainableBoostingClassifier

        optional_models["EBM"] = ExplainableBoostingClassifier
    except ImportError:
        print("interpret is not installed. Skipping EBM.")

    return optional_models


def build_model_configs(linear_preprocessor, tree_preprocessor):
    """Define models and memory-conscious GridSearchCV parameter grids."""
    model_configs = {
        "Logistic Regression": {
            "preprocessor": linear_preprocessor,
            "model": LogisticRegression(
                solver="liblinear",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "model__C": [0.1, 1.0, 10.0],
            },
        },
        "Random Forest": {
            "preprocessor": tree_preprocessor,
            "model": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=MODEL_N_JOBS,
            ),
            "param_grid": {
                "model__n_estimators": [100],
                "model__max_depth": [10, 20],
                "model__min_samples_leaf": [5, 20],
                "model__max_features": ["sqrt"],
            },
        },
    }

    optional_model_classes = get_optional_model_classes()

    if "XGBoost" in optional_model_classes:
        model_configs["XGBoost"] = {
            "preprocessor": tree_preprocessor,
            "model": optional_model_classes["XGBoost"](
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=MODEL_N_JOBS,
            ),
            "param_grid": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "model__min_child_weight": [1, 5],
            },
        }

    if "LightGBM" in optional_model_classes:
        model_configs["LightGBM"] = {
            "preprocessor": tree_preprocessor,
            "model": optional_model_classes["LightGBM"](
                objective="binary",
                random_state=RANDOM_STATE,
                n_jobs=MODEL_N_JOBS,
                verbose=-1,
                force_col_wise=True,
            ),
            "param_grid": {
                "model__n_estimators": [100],
                "model__num_leaves": [31, 63],
                "model__learning_rate": [0.05],
                "model__min_child_samples": [20, 100],
            },
        }

    if "EBM" in optional_model_classes:
        model_configs["EBM"] = {
            "preprocessor": tree_preprocessor,
            "model": optional_model_classes["EBM"](
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "model__interactions": [0, 5],
            },
        }

    return model_configs


def get_positive_class_scores(model, X):
    """Get prediction scores for TARGET = 1."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    raise ValueError("The model does not provide predict_proba or decision_function.")


def evaluate_model(model_name, model, X_valid, y_valid, best_params, cv_best_score, fit_seconds):
    """Evaluate one fitted model on the validation set."""
    y_pred = model.predict(X_valid)
    y_score = get_positive_class_scores(model, X_valid)

    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred, labels=[0, 1]).ravel()

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_valid, y_pred),
        "precision_target_1": precision_score(y_valid, y_pred, zero_division=0),
        "recall_target_1": recall_score(y_valid, y_pred, zero_division=0),
        "f1_target_1": f1_score(y_valid, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_valid, y_score),
        "pr_auc": average_precision_score(y_valid, y_score),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "cv_best_average_precision": cv_best_score,
        "fit_seconds": round(fit_seconds, 2),
        "best_params": pformat(best_params),
    }


def run_grid_search(model_name, config, X_train, y_train):
    """Run GridSearchCV for one model."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", config["preprocessor"]),
            ("model", config["model"]),
        ]
    )

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=config["param_grid"],
        scoring=SCORING,
        cv=cv,
        n_jobs=GRID_SEARCH_N_JOBS,
        refit=True,
        return_train_score=True,
        verbose=2,
    )

    print(f"\n=== Training {model_name} ===")
    print(f"Parameter grid: {config['param_grid']}")

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    fit_seconds = time.time() - start_time

    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.insert(0, "model", model_name)

    return grid_search, cv_results, fit_seconds


def save_csv_and_excel(results_df, best_params_df, best_model_df, cv_results_df):
    """Save result tables as CSV files and one Excel workbook."""
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    results_csv = OUTPUT_TABLES_DIR / "experiment_1_model_comparison.csv"
    best_params_csv = OUTPUT_TABLES_DIR / "experiment_1_best_parameters.csv"
    best_model_csv = OUTPUT_TABLES_DIR / "experiment_1_best_model_summary.csv"
    cv_results_csv = OUTPUT_TABLES_DIR / "experiment_1_grid_search_cv_results.csv"
    excel_path = OUTPUT_TABLES_DIR / "experiment_1_results.xlsx"

    results_df.to_csv(results_csv, index=False)
    best_params_df.to_csv(best_params_csv, index=False)
    best_model_df.to_csv(best_model_csv, index=False)
    cv_results_df.to_csv(cv_results_csv, index=False)

    try:
        with pd.ExcelWriter(excel_path) as writer:
            results_df.to_excel(writer, sheet_name="validation_metrics", index=False)
            best_params_df.to_excel(writer, sheet_name="best_parameters", index=False)
            best_model_df.to_excel(writer, sheet_name="best_model", index=False)
            cv_results_df.to_excel(writer, sheet_name="grid_search_cv", index=False)
        print(f"Saved Excel workbook: {excel_path}")
    except ImportError:
        print("Could not save Excel file. Please install openpyxl and run again.")

    print("\nSaved CSV files:")
    print(results_csv)
    print(best_params_csv)
    print(best_model_csv)
    print(cv_results_csv)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    print("Preparing data...")
    data = prepare_data()

    X_train = data["X_train"]
    X_valid = data["X_valid"]
    y_train = data["y_train"]
    y_valid = data["y_valid"]

    model_configs = build_model_configs(
        linear_preprocessor=data["linear_preprocessor"],
        tree_preprocessor=data["tree_preprocessor"],
    )

    validation_results = []
    best_parameter_rows = []
    cv_result_tables = []

    for model_name, config in model_configs.items():
        try:
            grid_search, cv_results, fit_seconds = run_grid_search(
                model_name=model_name,
                config=config,
                X_train=X_train,
                y_train=y_train,
            )

            validation_result = evaluate_model(
                model_name=model_name,
                model=grid_search.best_estimator_,
                X_valid=X_valid,
                y_valid=y_valid,
                best_params=grid_search.best_params_,
                cv_best_score=grid_search.best_score_,
                fit_seconds=fit_seconds,
            )

            validation_results.append(validation_result)
            best_parameter_rows.append(
                {
                    "model": model_name,
                    "best_cv_average_precision": grid_search.best_score_,
                    "best_params": pformat(grid_search.best_params_),
                }
            )
            cv_result_tables.append(cv_results)

            print(f"\nBest parameters for {model_name}:")
            print(grid_search.best_params_)
            print(f"Validation PR-AUC: {validation_result['pr_auc']:.4f}")

        except Exception as error:
            print(f"\n{model_name} failed and will be skipped.")
            print(f"Reason: {error}")

    if not validation_results:
        raise RuntimeError("No models completed successfully.")

    results_df = pd.DataFrame(validation_results).sort_values(
        by="pr_auc",
        ascending=False,
    )
    best_params_df = pd.DataFrame(best_parameter_rows)
    best_model_df = results_df.head(1).copy()
    cv_results_df = pd.concat(cv_result_tables, ignore_index=True)

    save_csv_and_excel(
        results_df=results_df,
        best_params_df=best_params_df,
        best_model_df=best_model_df,
        cv_results_df=cv_results_df,
    )

    print("\n=== Experiment 1 Model Comparison ===")
    print(results_df)

    print("\n=== Best Model By Validation PR-AUC ===")
    print(best_model_df)


if __name__ == "__main__":
    main()
