import argparse
from pathlib import Path
from math import prod
from pprint import pformat
import time
import traceback
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import (
    AllKNN,
    EditedNearestNeighbours,
    NeighbourhoodCleaningRule,
    RandomUnderSampler,
    TomekLinks,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline

from preprocessing import (
    RANDOM_STATE,
    add_financial_features,
    build_tree_preprocessor,
    get_feature_types,
    load_data,
    split_features_target,
    split_train_valid_test,
)


TABLES_DIR = Path("outputs/tables/experiment_2")
FIGURES_DIR = Path("outputs/figures/experiment_2")
QUICK_CM_DIR = FIGURES_DIR / "quick_test_confusion_matrices"
FULL_CM_DIR = FIGURES_DIR / "confusion_matrices"
FULL_PR_DIR = FIGURES_DIR / "pr_curves"
FULL_ROC_DIR = FIGURES_DIR / "roc_curves"

QUICK_SAMPLE_SIZE = 10000
N_JOBS = 4
SLOW_METHODS = {
    "TomekLinks",
    "EditedNearestNeighbours",
    "AllKNN",
    "NeighbourhoodCleaningRule",
    "SMOTETomek",
    "SMOTEENN",
}
THRESHOLD_STRATEGIES = [
    "threshold_0_5",
    "best_f1_threshold",
    "best_cost_threshold_fn5",
    "best_cost_threshold_fn10",
    "best_recall_min_precision_0_20",
]


class RecordingSampler(BaseEstimator):
    """Wrap a sampler and record the class distribution from fit_resample."""

    def __init__(self, sampler):
        self.sampler = sampler
        self.sampling_summary_ = None

    def fit_resample(self, X, y):
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        self.sampling_summary_ = summarize_class_distribution(
            y_resampled,
            "train_sampled",
        )
        self.sampling_summary_["sampler_applied"] = True
        self.sampling_summary_["sampling_summary_status"] = "recorded_during_fit"
        return X_resampled, y_resampled


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 2: compare imbalanced-data handling strategies."
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Use a 10000-row stratified sample and run every planned method once.",
    )
    parser.add_argument(
        "--include-slow-methods",
        action="store_true",
        help="In full mode, include slow methods such as AllKNN and SMOTEENN.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional list of method names to run, for example: SMOTE BorderlineSMOTE ADASYN.",
    )
    parser.add_argument(
        "--include-ebm",
        action="store_true",
        help="Optionally also run EBM_Original as an interpretable comparison model.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from existing output files and skip completed or failed methods.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional name used in output filenames to avoid overwriting old results.",
    )
    return parser.parse_args()


def get_xgboost_classifier(quick_test=False):
    from xgboost import XGBClassifier

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "verbosity": 0,
    }
    if quick_test:
        params.update({"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1})

    return XGBClassifier(**params)


def get_ebm_classifier(quick_test=False):
    from interpret.glassbox import ExplainableBoostingClassifier

    max_bins = 64 if quick_test else 128
    max_rounds = 1000 if quick_test else 3000
    return ExplainableBoostingClassifier(
        random_state=RANDOM_STATE,
        interactions=0,
        max_bins=max_bins,
        max_rounds=max_rounds,
    )


def get_xgboost_param_grid(quick_test=False):
    if quick_test:
        return {
            "model__n_estimators": [50],
            "model__max_depth": [3],
            "model__learning_rate": [0.1],
            "model__reg_alpha": [0],
            "model__reg_lambda": [1],
            "model__gamma": [0],
        }

    return {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8],
        "model__colsample_bytree": [0.8],
        "model__min_child_weight": [1],
        "model__reg_alpha": [0, 0.1],
        "model__reg_lambda": [1, 5],
        "model__gamma": [0, 1],
    }


def get_scale_pos_weight_grid(y_train, quick_test=False):
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    if quick_test:
        values = [scale_pos_weight]
    else:
        values = [
            max(1, scale_pos_weight - 2),
            scale_pos_weight,
            scale_pos_weight + 2,
        ]
    values = [round(float(value), 4) for value in values]
    return {"model__scale_pos_weight": values}


def get_ebm_param_grid(quick_test=False):
    if quick_test:
        return {"model__interactions": [0]}
    return {"model__interactions": [0, 5]}


def create_stratified_quick_sample(df):
    sample_size = min(QUICK_SAMPLE_SIZE, len(df))
    _, sample_df = train_test_split(
        df,
        test_size=sample_size,
        stratify=df["TARGET"],
        random_state=RANDOM_STATE,
    )
    return sample_df.reset_index(drop=True)


def prepare_experiment_data(quick_test=False):
    df = load_data()
    if quick_test:
        df = create_stratified_quick_sample(df)

    df = add_financial_features(df)
    X, y = split_features_target(df)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_train_valid_test(X, y)

    numeric_features, categorical_features = get_feature_types(X_train)
    tree_preprocessor = build_tree_preprocessor(numeric_features, categorical_features)

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "tree_preprocessor": tree_preprocessor,
    }


def calculate_scale_pos_weight(y_train):
    class_counts = y_train.value_counts()
    negative_count = class_counts.get(0, 0)
    positive_count = class_counts.get(1, 0)
    if positive_count == 0:
        raise ValueError("No positive TARGET=1 rows found in training data.")
    return negative_count / positive_count


def build_sampler(method_name):
    samplers = {
        "RandomOverSampler": RandomOverSampler(random_state=RANDOM_STATE),
        "SMOTE": SMOTE(random_state=RANDOM_STATE, k_neighbors=3),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=3),
        "ADASYN": ADASYN(random_state=RANDOM_STATE, n_neighbors=3),
        "RandomUnderSampler": RandomUnderSampler(random_state=RANDOM_STATE),
        "TomekLinks": TomekLinks(n_jobs=N_JOBS),
        "EditedNearestNeighbours": EditedNearestNeighbours(n_jobs=N_JOBS),
        "AllKNN": AllKNN(n_jobs=N_JOBS),
        "NeighbourhoodCleaningRule": NeighbourhoodCleaningRule(n_jobs=N_JOBS),
        "SMOTETomek": SMOTETomek(
            random_state=RANDOM_STATE,
            smote=SMOTE(random_state=RANDOM_STATE, k_neighbors=3),
            tomek=TomekLinks(n_jobs=N_JOBS),
        ),
        "SMOTEENN": SMOTEENN(
            random_state=RANDOM_STATE,
            smote=SMOTE(random_state=RANDOM_STATE, k_neighbors=3),
            enn=EditedNearestNeighbours(n_jobs=N_JOBS),
        ),
    }
    return samplers[method_name]


def build_method_configs(y_train, quick_test=False, include_ebm=False):
    sampler_methods = [
        "RandomOverSampler",
        "SMOTE",
        "BorderlineSMOTE",
        "ADASYN",
        "RandomUnderSampler",
        "TomekLinks",
        "EditedNearestNeighbours",
        "AllKNN",
        "NeighbourhoodCleaningRule",
        "SMOTETomek",
        "SMOTEENN",
    ]

    configs = {
        "Original": {
            "model_family": "XGBoost",
            "sampler": None,
            "model": get_xgboost_classifier(quick_test=quick_test),
            "param_grid": get_xgboost_param_grid(quick_test=quick_test),
        },
        "XGBoost_scale_pos_weight": {
            "model_family": "XGBoost",
            "sampler": None,
            "model": get_xgboost_classifier(quick_test=quick_test),
            "param_grid": {
                **get_xgboost_param_grid(quick_test=quick_test),
                **get_scale_pos_weight_grid(y_train, quick_test=quick_test),
            },
        },
    }

    for method_name in sampler_methods:
        configs[method_name] = {
            "model_family": "XGBoost",
            "sampler": build_sampler(method_name),
            "model": get_xgboost_classifier(quick_test=quick_test),
            "param_grid": get_xgboost_param_grid(quick_test=quick_test),
        }

    if include_ebm:
        try:
            configs["EBM_Original"] = {
                "model_family": "EBM",
                "sampler": None,
                "model": get_ebm_classifier(quick_test=quick_test),
                "param_grid": get_ebm_param_grid(quick_test=quick_test),
            }
        except ImportError:
            print("EBM is not available. Skipping EBM_Original.")

    return configs


def select_methods(configs, args):
    selected = list(configs.keys())

    if args.methods:
        unknown_methods = sorted(set(args.methods) - set(configs))
        if unknown_methods:
            raise ValueError(f"Unknown methods: {unknown_methods}")
        selected = args.methods
    elif not args.quick_test and not args.include_slow_methods:
        selected = [method for method in selected if method not in SLOW_METHODS]

    return {method: configs[method] for method in selected}


def build_pipeline(preprocessor, sampler, model):
    if sampler is None:
        return SklearnPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    return ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("sampler", RecordingSampler(sampler)),
            ("model", model),
        ]
    )


def run_grid_search(method_name, config, data, quick_test=False):
    pipeline = build_pipeline(
        preprocessor=data["tree_preprocessor"],
        sampler=config["sampler"],
        model=config["model"],
    )

    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
    }
    cv_splits = 2 if quick_test else 3
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=config["param_grid"],
        scoring=scoring,
        refit="pr_auc",
        cv=cv,
        n_jobs=1,
        verbose=2,
        return_train_score=False,
        error_score="raise",
    )

    print(f"\n=== Running {method_name} ===")
    print(f"Model family: {config['model_family']}")
    print(f"Sampler: {type(config['sampler']).__name__ if config['sampler'] else 'None'}")
    print(f"Parameter grid: {config['param_grid']}")
    candidate_count = prod(len(values) for values in config["param_grid"].values())
    print(f"Grid candidates: {candidate_count}")
    print(f"Total CV fits: {candidate_count * cv_splits}")

    start_time = time.time()
    search.fit(data["X_train"], data["y_train"])
    fit_seconds = time.time() - start_time

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.insert(0, "method", method_name)
    cv_results.insert(1, "model_family", config["model_family"])

    return search, cv_results, fit_seconds


def summarize_class_distribution(y_values, prefix):
    counts = pd.Series(y_values).value_counts()
    count_0 = int(counts.get(0, 0))
    count_1 = int(counts.get(1, 0))
    total = count_0 + count_1

    return {
        f"{prefix}_count_0": count_0,
        f"{prefix}_count_1": count_1,
        f"{prefix}_total": total,
        f"{prefix}_percent_0": round(count_0 / total * 100, 4) if total else 0,
        f"{prefix}_percent_1": round(count_1 / total * 100, 4) if total else 0,
        f"{prefix}_imbalance_ratio_0_to_1": round(count_0 / count_1, 4)
        if count_1
        else None,
    }


def get_sampling_summary(method_name, best_estimator, X_train, y_train):
    """Count TARGET classes after the selected sampler is applied."""
    original_summary = summarize_class_distribution(y_train, "train_original")

    if "sampler" not in best_estimator.named_steps:
        sampled_summary = summarize_class_distribution(y_train, "train_sampled")
        sampled_summary["sampler_applied"] = False
        sampled_summary["sampling_summary_status"] = "no_sampler"
        return {**original_summary, **sampled_summary}

    sampler = best_estimator.named_steps["sampler"]
    sampled_summary = getattr(sampler, "sampling_summary_", None)

    if sampled_summary is None:
        sampled_summary = {
            "train_sampled_count_0": None,
            "train_sampled_count_1": None,
            "train_sampled_total": None,
            "train_sampled_percent_0": None,
            "train_sampled_percent_1": None,
            "train_sampled_imbalance_ratio_0_to_1": None,
            "sampler_applied": True,
            "sampling_summary_status": "not_available",
        }

    return {**original_summary, **sampled_summary}


def get_positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise ValueError("Model does not provide predict_proba or decision_function.")


def find_thresholds(y_true, y_score):
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    thresholds = {
        "threshold_0_5": 0.5,
    }

    if len(pr_thresholds) > 0:
        f1_values = 2 * precision[:-1] * recall[:-1] / (
            precision[:-1] + recall[:-1] + 1e-12
        )
        thresholds["best_f1_threshold"] = pr_thresholds[f1_values.argmax()]

        thresholds["best_cost_threshold_fn5"] = find_best_cost_threshold(
            y_true,
            y_score,
            fn_cost=5,
            fp_cost=1,
        )
        thresholds["best_cost_threshold_fn10"] = find_best_cost_threshold(
            y_true,
            y_score,
            fn_cost=10,
            fp_cost=1,
        )

        valid_precision_mask = precision[:-1] >= 0.20
        if valid_precision_mask.any():
            recall_candidates = recall[:-1].copy()
            recall_candidates[~valid_precision_mask] = -1
            thresholds["best_recall_min_precision_0_20"] = pr_thresholds[
                recall_candidates.argmax()
            ]
        else:
            thresholds["best_recall_min_precision_0_20"] = thresholds[
                "best_f1_threshold"
            ]

    return thresholds


def find_best_cost_threshold(y_true, y_score, fn_cost, fp_cost):
    candidate_thresholds = pd.Series(y_score).drop_duplicates().sort_values().to_numpy()
    if len(candidate_thresholds) > 500:
        candidate_thresholds = pd.Series(candidate_thresholds).quantile(
            [i / 500 for i in range(501)]
        ).drop_duplicates().to_numpy()

    best_threshold = 0.5
    best_cost = None

    for threshold in candidate_thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = fn_cost * fn + fp_cost * fp
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    return best_threshold


def evaluate_threshold(method_name, model_family, y_true, y_score, threshold_name, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "method": method_name,
        "model_family": model_family,
        "threshold_strategy": threshold_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_target_1": precision_score(y_true, y_pred, zero_division=0),
        "recall_target_1": recall_score(y_true, y_pred, zero_division=0),
        "f1_target_1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn,
        "cost_fn5_fp1": 5 * fn + fp,
        "cost_fn10_fp1": 10 * fn + fp,
    }


def evaluate_model(method_name, model_family, estimator, X_valid, y_valid):
    y_score = get_positive_scores(estimator, X_valid)
    thresholds = find_thresholds(y_valid, y_score)
    rows = []
    for threshold_name in THRESHOLD_STRATEGIES:
        if threshold_name in thresholds:
            rows.append(
                evaluate_threshold(
                    method_name,
                    model_family,
                    y_valid,
                    y_score,
                    threshold_name,
                    thresholds[threshold_name],
                )
            )
    return rows, y_score


def plot_confusion_matrix(row, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(image, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(f"{row['method']} - {row['threshold_strategy']}")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i][j], ha="center", va="center", color="black")

    fig.tight_layout()
    file_name = f"{safe_file_name(row['method'])}_{row['threshold_strategy']}.png"
    fig.savefig(output_dir / file_name, dpi=150)
    plt.close(fig)


def plot_pr_curve(method_name, y_valid, y_score, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_valid, y_score)
    pr_auc = average_precision_score(y_valid, y_score)

    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"PR-AUC={pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve - {method_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{safe_file_name(method_name)}.png", dpi=150)
    plt.close()


def plot_roc_curve(method_name, y_valid, y_score, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_valid, y_score)
    roc_auc = roc_auc_score(y_valid, y_score)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC-AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {method_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{safe_file_name(method_name)}.png", dpi=150)
    plt.close()


def safe_file_name(value):
    return str(value).replace(" ", "_").replace("/", "_")


def build_output_name(base_name, run_name=None):
    if not run_name:
        return base_name
    return f"{base_name}_{run_name}"


def get_output_paths(quick_test=False, run_name=None):
    if quick_test:
        base_name = build_output_name("experiment_2_quick_test", run_name)
        return {
            "results": TABLES_DIR / f"{base_name}_results.csv",
            "failures": TABLES_DIR / f"{base_name}_failures.csv",
            "grid": TABLES_DIR / f"{base_name}_grid_search_results.csv",
        }

    base_name = build_output_name("experiment_2", run_name)
    return {
        "results": TABLES_DIR / f"{base_name}_results.csv",
        "failures": TABLES_DIR / f"{base_name}_failures.csv",
        "grid": TABLES_DIR / f"{base_name}_grid_search_results.csv",
    }


def save_outputs(results, failures, cv_tables, quick_test=False, run_name=None, quiet=False):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    failures_df = pd.DataFrame(failures)
    cv_results_df = pd.concat(cv_tables, ignore_index=True) if cv_tables else pd.DataFrame()
    output_paths = get_output_paths(quick_test=quick_test, run_name=run_name)

    results_df.to_csv(output_paths["results"], index=False)
    cv_results_df.to_csv(output_paths["grid"], index=False)

    if failures:
        failures_df.to_csv(output_paths["failures"], index=False)

    if not quiet:
        print(f"\nSaved results: {output_paths['results']}")
        print(f"Saved grid search results: {output_paths['grid']}")
        if failures:
            print(f"Saved failures: {output_paths['failures']}")

    return results_df, failures_df, cv_results_df


def load_existing_progress(quick_test=False, run_name=None):
    output_paths = get_output_paths(quick_test=quick_test, run_name=run_name)
    results = []
    failures = []
    cv_tables = []

    if output_paths["results"].exists() and output_paths["results"].stat().st_size > 0:
        results_df = pd.read_csv(output_paths["results"])
        results = results_df.to_dict("records")

    if output_paths["failures"].exists() and output_paths["failures"].stat().st_size > 0:
        failures_df = pd.read_csv(output_paths["failures"])
        failures = failures_df.to_dict("records")

    if output_paths["grid"].exists() and output_paths["grid"].stat().st_size > 0:
        cv_tables.append(pd.read_csv(output_paths["grid"]))

    completed_methods = {row["method"] for row in results if "method" in row}
    failed_methods = {row["method"] for row in failures if "method" in row}

    return results, failures, cv_tables, completed_methods | failed_methods


def run_experiment(args):
    warnings.filterwarnings("ignore", category=UserWarning)

    data = prepare_experiment_data(quick_test=args.quick_test)
    all_configs = build_method_configs(
        y_train=data["y_train"],
        quick_test=args.quick_test,
        include_ebm=args.include_ebm,
    )
    method_configs = select_methods(all_configs, args)

    print("\n=== Experiment 2 Setup ===")
    print(f"Quick test: {args.quick_test}")
    print(f"Methods: {list(method_configs.keys())}")
    print(f"X_train: {data['X_train'].shape}")
    print(f"X_valid: {data['X_valid'].shape}")
    print(f"n_jobs limit: {N_JOBS}")
    print(f"Run name: {args.run_name if args.run_name else 'default'}")

    if args.resume:
        results, failures, cv_tables, completed_or_failed_methods = load_existing_progress(
            quick_test=args.quick_test,
            run_name=args.run_name,
        )
        print(f"Resume enabled. Skipping methods already recorded: {sorted(completed_or_failed_methods)}")
    else:
        results = []
        failures = []
        cv_tables = []
        completed_or_failed_methods = set()

    cm_dir = QUICK_CM_DIR if args.quick_test else FULL_CM_DIR

    for method_name, config in method_configs.items():
        if method_name in completed_or_failed_methods:
            print(f"\nSkipping {method_name} because it is already recorded.")
            continue

        try:
            search, cv_results, fit_seconds = run_grid_search(
                method_name,
                config,
                data,
                quick_test=args.quick_test,
            )
            cv_tables.append(cv_results)

            method_rows, y_score = evaluate_model(
                method_name=method_name,
                model_family=config["model_family"],
                estimator=search.best_estimator_,
                X_valid=data["X_valid"],
                y_valid=data["y_valid"],
            )
            sampling_summary = get_sampling_summary(
                method_name=method_name,
                best_estimator=search.best_estimator_,
                X_train=data["X_train"],
                y_train=data["y_train"],
            )

            for row in method_rows:
                row["best_cv_pr_auc"] = search.best_score_
                row["fit_seconds"] = round(fit_seconds, 2)
                row["best_params"] = pformat(search.best_params_)
                row.update(sampling_summary)
                results.append(row)
                if row["threshold_strategy"] == "threshold_0_5":
                    plot_confusion_matrix(row, cm_dir)

            if not args.quick_test:
                plot_pr_curve(method_name, data["y_valid"], y_score, FULL_PR_DIR)
                plot_roc_curve(method_name, data["y_valid"], y_score, FULL_ROC_DIR)

            print(f"Finished {method_name}. Best CV PR-AUC: {search.best_score_:.4f}")
            save_outputs(
                results=results,
                failures=failures,
                cv_tables=cv_tables,
                quick_test=args.quick_test,
                run_name=args.run_name,
                quiet=True,
            )
            print("Progress saved after this method.")

        except KeyboardInterrupt:
            save_outputs(
                results=results,
                failures=failures,
                cv_tables=cv_tables,
                quick_test=args.quick_test,
                run_name=args.run_name,
                quiet=True,
            )
            print("\nInterrupted by user. Progress saved before stopping.")
            raise

        except Exception as error:
            failures.append(
                {
                    "method": method_name,
                    "model_family": config["model_family"],
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"\n{method_name} failed and will be skipped.")
            print(f"Reason: {error}")
            save_outputs(
                results=results,
                failures=failures,
                cv_tables=cv_tables,
                quick_test=args.quick_test,
                run_name=args.run_name,
                quiet=True,
            )
            print("Failure log saved. Continuing if more methods remain.")

    results_df, _, _ = save_outputs(
        results=results,
        failures=failures,
        cv_tables=cv_tables,
        quick_test=args.quick_test,
        run_name=args.run_name,
    )

    if not results_df.empty:
        print("\n=== Top Results By PR-AUC ===")
        display_columns = [
            "method",
            "threshold_strategy",
            "precision_target_1",
            "recall_target_1",
            "f1_target_1",
            "pr_auc",
            "fn",
            "fp",
            "cost_fn5_fp1",
            "cost_fn10_fp1",
        ]
        print(
            results_df.sort_values("pr_auc", ascending=False)[display_columns]
            .head(20)
            .to_string(index=False)
        )


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
