import argparse
import ast
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from preprocessing import (
    RANDOM_STATE,
    add_financial_features,
    build_tree_preprocessor,
    get_feature_types,
    load_data,
    split_features_target,
    split_train_valid_test,
)


TABLES_DIR = Path("outputs/tables/experiment_3")
FIGURES_DIR = Path("outputs/figures/experiment_3")
MODELS_DIR = Path("outputs/models")
EXPERIMENT_2_RESULTS_PATH = Path(
    "outputs/tables/experiment_2/experiment_2_full_all_v2_results.csv"
)

QUICK_SAMPLE_SIZE = 10000
QUICK_SHAP_SAMPLE_SIZE = 1000
FULL_SHAP_SAMPLE_SIZE = 4000
N_JOBS = 4

IMPORTANT_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "EXT_SOURCE_MEAN",
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "AMT_ANNUITY",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_GOODS_RATIO",
    "DAYS_BIRTH",
    "AGE_YEARS",
    "DAYS_EMPLOYED",
    "EMPLOYED_YEARS",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "OCCUPATION_TYPE",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]

SHAP_DEPENDENCE_FEATURES = [
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "AGE_YEARS",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 3: XGBoost, SHAP, and EBM interpretability analysis."
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Use a smaller stratified sample and only generate the fastest outputs.",
    )
    return parser.parse_args()


def create_stratified_sample(df, sample_size):
    """Take a reproducible stratified sample while preserving class imbalance."""
    sample_size = min(sample_size, len(df))
    _, sample_df = train_test_split(
        df,
        test_size=sample_size,
        stratify=df["TARGET"],
        random_state=RANDOM_STATE,
    )
    return sample_df.reset_index(drop=True)


def prepare_experiment_data(quick_test=False):
    """Load data, add financial features, and create stratified splits."""
    df = load_data()
    if quick_test:
        df = create_stratified_sample(df, QUICK_SAMPLE_SIZE)

    df = add_financial_features(df)
    X, y = split_features_target(df)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_train_valid_test(X, y)
    numeric_features, categorical_features = get_feature_types(X_train)

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }


def calculate_scale_pos_weight(y_train):
    class_counts = y_train.value_counts()
    negative_count = class_counts.get(0, 0)
    positive_count = class_counts.get(1, 0)
    if positive_count == 0:
        raise ValueError("No TARGET=1 rows found in training data.")
    return float(negative_count / positive_count)


def load_experiment_2_best_xgboost_params():
    """Reuse practical XGBoost hyperparameters from Experiment 2 when available."""
    if not EXPERIMENT_2_RESULTS_PATH.exists():
        return {}

    try:
        results = pd.read_csv(EXPERIMENT_2_RESULTS_PATH)
        xgb_results = results[results["model_family"].eq("XGBoost")].copy()
        if xgb_results.empty:
            return {}

        # Prefer the row that minimized FN-heavy cost, then reuse only model params.
        best_row = xgb_results.sort_values("cost_fn10_fp1").iloc[0]
        raw_params = ast.literal_eval(best_row["best_params"])
        params = {}
        for key, value in raw_params.items():
            if key.startswith("model__"):
                params[key.replace("model__", "")] = value
        return params
    except Exception as exc:
        print(f"Could not reuse Experiment 2 XGBoost params: {exc}")
        return {}


def build_xgboost_pipeline(data, quick_test=False):
    from xgboost import XGBClassifier

    numeric_features = data["numeric_features"]
    categorical_features = data["categorical_features"]
    preprocessor = build_tree_preprocessor(numeric_features, categorical_features)
    scale_pos_weight = calculate_scale_pos_weight(data["y_train"])

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "verbosity": 0,
        "scale_pos_weight": scale_pos_weight,
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "gamma": 0,
    }

    if not quick_test:
        params.update(load_experiment_2_best_xgboost_params())
        params["scale_pos_weight"] = scale_pos_weight
    else:
        params.update(
            {
                "n_estimators": 50,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            }
        )

    model = XGBClassifier(**params)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline, params


def get_feature_names(preprocessor):
    """Return readable feature names after one-hot preprocessing."""
    try:
        return preprocessor.get_feature_names_out()
    except AttributeError:
        numeric_names = preprocessor.transformers_[0][2]
        categorical_pipeline = preprocessor.named_transformers_["categorical"]
        encoder = categorical_pipeline.named_steps["one_hot"]
        categorical_names = encoder.get_feature_names_out(preprocessor.transformers_[1][2])
        return np.array(list(numeric_names) + list(categorical_names))


def clean_feature_name(feature_name):
    """Remove ColumnTransformer prefixes for easier reading."""
    return feature_name.replace("numeric__", "").replace("categorical__", "")


def original_feature_from_processed(processed_feature_name):
    """Map one-hot names back to the original feature when possible."""
    cleaned = clean_feature_name(processed_feature_name)
    for feature in sorted(IMPORTANT_FEATURES, key=len, reverse=True):
        if cleaned == feature or cleaned.startswith(f"{feature}_"):
            return feature
    return cleaned


def save_xgboost_feature_importance(pipeline):
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = [clean_feature_name(name) for name in get_feature_names(preprocessor)]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "original_feature": [
                original_feature_from_processed(name) for name in feature_names
            ],
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLES_DIR / "experiment_3_xgboost_feature_importance.csv"
    importance_df.to_csv(output_path, index=False)

    top_features = importance_df.head(20).sort_values("importance")
    plt.figure(figsize=(9, 7))
    plt.barh(top_features["feature"], top_features["importance"], color="#2b6cb0")
    plt.xlabel("XGBoost feature importance")
    plt.title("Experiment 3: XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "experiment_3_xgboost_feature_importance.png", dpi=150)
    plt.close()

    return importance_df


def select_shap_sample(X_test, y_test, sample_size):
    sample_size = min(sample_size, len(X_test))
    if sample_size == len(X_test):
        return X_test.copy(), y_test.copy()

    _, X_sample, _, y_sample = train_test_split(
        X_test,
        y_test,
        test_size=sample_size,
        stratify=y_test,
        random_state=RANDOM_STATE,
    )
    return X_sample.copy(), y_sample.copy()


def run_shap_analysis(pipeline, data, quick_test=False):
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost is not installed. Skipping SHAP analysis.")
        return None

    try:
        import shap
    except ImportError:
        shap = None
        print("SHAP package is not installed. Saving SHAP importance table and bar plot only.")

    sample_size = QUICK_SHAP_SAMPLE_SIZE if quick_test else FULL_SHAP_SAMPLE_SIZE
    X_sample, _ = select_shap_sample(data["X_test"], data["y_test"], sample_size)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_sample_processed = preprocessor.transform(X_sample)
    feature_names = [clean_feature_name(name) for name in get_feature_names(preprocessor)]

    if hasattr(X_sample_processed, "toarray"):
        X_sample_processed = X_sample_processed.toarray()

    # XGBoost 3.x can store base_score in a format that some SHAP versions cannot
    # parse through TreeExplainer(model). pred_contribs uses XGBoost's native Tree
    # SHAP implementation and avoids that version-specific base_score issue.
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X_sample_processed, feature_names=feature_names)
    shap_values = booster.predict(dmatrix, pred_contribs=True)[:, :-1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "original_feature": [
                original_feature_from_processed(name) for name in feature_names
            ],
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    shap_importance_df.to_csv(
        TABLES_DIR / "experiment_3_shap_feature_importance.csv",
        index=False,
    )

    plot_shap_bar(shap_importance_df)

    if shap is not None:
        save_shap_beeswarm_plot(
            shap,
            shap_values,
            X_sample_processed,
            feature_names,
            max_display=20,
            output_name="experiment_3_shap_summary_beeswarm.png",
        )
        save_shap_beeswarm_plot(
            shap,
            shap_values,
            X_sample_processed,
            feature_names,
            max_display=10,
            output_name="experiment_3_shap_summary_beeswarm_top10.png",
        )

    if shap is not None and not quick_test:
        feature_index = {name: index for index, name in enumerate(feature_names)}
        for feature in SHAP_DEPENDENCE_FEATURES:
            if feature in feature_index:
                shap.dependence_plot(
                    feature,
                    shap_values,
                    X_sample_processed,
                    feature_names=feature_names,
                    interaction_index=None,
                    show=False,
                )
                plt.tight_layout()
                plt.savefig(
                    FIGURES_DIR / f"experiment_3_shap_dependence_{feature}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()

    return shap_importance_df


def save_shap_beeswarm_plot(
    shap,
    shap_values,
    X_sample_processed,
    feature_names,
    max_display,
    output_name,
):
    """Save a SHAP beeswarm plot with enough margins for long feature names."""
    plot_width = 14 if max_display <= 10 else 16
    plot_height = 6 if max_display <= 10 else 8

    shap.summary_plot(
        shap_values,
        X_sample_processed,
        feature_names=feature_names,
        show=False,
        max_display=max_display,
        plot_size=(plot_width, plot_height),
    )
    fig = plt.gcf()
    fig.subplots_adjust(left=0.46, right=0.92, bottom=0.16, top=0.96)
    plt.savefig(FIGURES_DIR / output_name, dpi=150)
    plt.close()


def plot_shap_bar(shap_importance_df):
    """Save a readable SHAP bar plot without clipped labels or axis text."""
    top_features = shap_importance_df.head(20).sort_values("mean_abs_shap")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(top_features["feature"], top_features["mean_abs_shap"], color="#1f77b4")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Experiment 3: SHAP Feature Importance")

    max_value = top_features["mean_abs_shap"].max()
    ax.set_xlim(0, max_value * 1.18 if max_value > 0 else 1)

    for index, value in enumerate(top_features["mean_abs_shap"]):
        ax.text(
            value + max_value * 0.015,
            index,
            f"{value:.3f}",
            va="center",
            fontsize=8,
        )

    fig.subplots_adjust(left=0.38, right=0.97, bottom=0.12, top=0.92)
    fig.savefig(FIGURES_DIR / "experiment_3_shap_bar.png", dpi=150)
    plt.close(fig)



def build_ebm_preprocessor(numeric_features, categorical_features):
    """Preprocess data for EBM while preserving one column per original feature."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        verbose_feature_names_out=False,
    )


def run_ebm_analysis(data, quick_test=False):
    if quick_test:
        print("Quick-test mode skips EBM training to keep the smoke test fast.")
        return None

    try:
        from interpret.glassbox import ExplainableBoostingClassifier
    except ImportError:
        print("interpret is not installed. Skipping EBM analysis.")
        return None

    preprocessor = build_ebm_preprocessor(
        data["numeric_features"],
        data["categorical_features"],
    )
    feature_names = list(data["numeric_features"]) + list(data["categorical_features"])
    model = ExplainableBoostingClassifier(
        feature_names=feature_names,
        random_state=RANDOM_STATE,
        interactions=0,
        max_bins=128,
        max_rounds=1500,
        n_jobs=N_JOBS,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("\n=== Training EBM interpretable comparison model ===")
    pipeline.fit(data["X_train"], data["y_train"])

    ebm_model = pipeline.named_steps["model"]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": ebm_model.term_importances(),
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(
        TABLES_DIR / "experiment_3_ebm_feature_importance.csv",
        index=False,
    )

    top_features = importance_df.head(20).sort_values("importance")
    plt.figure(figsize=(9, 7))
    plt.barh(top_features["feature"], top_features["importance"], color="#2f855a")
    plt.xlabel("EBM term importance")
    plt.title("Experiment 3: EBM Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "experiment_3_ebm_feature_importance.png", dpi=150)
    plt.close()

    save_ebm_global_term_plot(
        ebm_model,
        feature_name="EXT_SOURCE_MEAN",
        output_path=FIGURES_DIR
        / "experiment_3_ebm_global_ext_source_mean_highres.png",
    )
    save_ebm_local_examples(pipeline, data, feature_names)
    return importance_df


def get_ebm_term_index(ebm_model, feature_name):
    """Find a single-feature EBM term by name."""
    term_names = getattr(ebm_model, "term_names_", None)
    if term_names is None:
        term_names = getattr(ebm_model, "term_names", None)
    if term_names is None:
        term_names = getattr(ebm_model, "feature_names_in_", None)
    if term_names is None:
        term_names = getattr(ebm_model, "feature_names", None)

    if term_names is None:
        raise ValueError("Could not find feature names on the fitted EBM model.")

    term_names = list(term_names)
    if feature_name not in term_names:
        raise ValueError(f"{feature_name} was not found in EBM terms.")
    return term_names.index(feature_name)


def make_interval_x_values(names, scores):
    """Convert EBM bin edges or labels into plottable x positions."""
    names = np.asarray(names)
    scores = np.asarray(scores)

    if len(names) == len(scores) + 1:
        edges = names.astype(float)
        x_values = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)
        return x_values, widths

    x_values = names.astype(float)
    if len(x_values) > 1:
        median_width = np.median(np.diff(np.sort(x_values)))
    else:
        median_width = 0.05
    widths = np.full(len(x_values), median_width)
    return x_values, widths


def save_ebm_global_term_plot(ebm_model, feature_name, output_path, dpi=300):
    """Save a high-resolution EBM global effect plot for one feature."""
    term_index = get_ebm_term_index(ebm_model, feature_name)
    term_data = ebm_model.explain_global().data(term_index)

    x_values, widths = make_interval_x_values(term_data["names"], term_data["scores"])
    scores = np.asarray(term_data["scores"], dtype=float)
    lower_bounds = term_data.get("lower_bounds")
    upper_bounds = term_data.get("upper_bounds")

    fig, (effect_ax, density_ax) = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    effect_ax.plot(x_values, scores, color="#1f77b4", linewidth=2.4)
    if lower_bounds is not None and upper_bounds is not None:
        effect_ax.fill_between(
            x_values,
            np.asarray(lower_bounds, dtype=float),
            np.asarray(upper_bounds, dtype=float),
            color="#1f77b4",
            alpha=0.16,
            linewidth=0,
        )
    effect_ax.axhline(0, color="#444444", linewidth=1, linestyle="--", alpha=0.7)
    effect_ax.set_title(
        f"EBM Global Explanation: {feature_name}",
        fontsize=16,
        pad=14,
    )
    effect_ax.set_ylabel("EBM score contribution", fontsize=12)
    effect_ax.grid(True, alpha=0.25)
    x_ticks = np.arange(0.0, 0.81, 0.2)
    effect_ax.set_xticks(x_ticks)
    effect_ax.tick_params(axis="x", labelbottom=True)

    density = term_data.get("density", {})
    density_names = density.get("names")
    density_scores = density.get("scores")
    if density_names is not None and density_scores is not None:
        density_x, density_widths = make_interval_x_values(density_names, density_scores)
        density_ax.bar(
            density_x,
            density_scores,
            width=density_widths,
            color="#ff7f0e",
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
        )
    density_ax.set_xlabel(feature_name, fontsize=12)
    density_ax.set_ylabel("Count", fontsize=12)
    density_ax.set_xticks(x_ticks)
    density_ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def get_local_contributions(ebm_model, transformed_row, feature_names, top_n=10):
    local_explanation = ebm_model.explain_local(transformed_row)
    local_data = local_explanation.data(0)
    names = local_data.get("names", feature_names)
    scores = local_data.get("scores", [])
    values = local_data.get("values", [])

    rows = []
    for name, score, value in zip(names, scores, values):
        rows.append(
            {
                "feature": name,
                "value": value,
                "contribution": score,
                "abs_contribution": abs(score),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_contribution", ascending=False).head(top_n)


def save_ebm_local_examples(pipeline, data, feature_names):
    X_test = data["X_test"]
    y_test = data["y_test"]
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    candidates = {
        "correct_target_0": (y_test.eq(0)) & (predictions == 0),
        "correct_target_1": (y_test.eq(1)) & (predictions == 1),
        "false_positive": (y_test.eq(0)) & (predictions == 1),
        "false_negative": (y_test.eq(1)) & (predictions == 0),
    }

    rows = []
    transformed_test = pipeline.named_steps["preprocessor"].transform(X_test)
    ebm_model = pipeline.named_steps["model"]

    for case_name, mask in candidates.items():
        indices = y_test[mask].index.tolist()
        if not indices:
            continue
        original_index = indices[0]
        position = X_test.index.get_loc(original_index)
        transformed_row = transformed_test[position : position + 1]
        contribution_df = get_local_contributions(
            ebm_model,
            transformed_row,
            feature_names,
            top_n=10,
        )
        for rank, row in enumerate(contribution_df.to_dict("records"), start=1):
            rows.append(
                {
                    "case": case_name,
                    "row_index": original_index,
                    "true_label": int(y_test.loc[original_index]),
                    "predicted_probability": probabilities[position],
                    "predicted_class_0_5": int(predictions[position]),
                    "rank": rank,
                    **row,
                }
            )

    if rows:
        pd.DataFrame(rows).to_csv(
            TABLES_DIR / "experiment_3_ebm_local_explanations.csv",
            index=False,
        )


def summarize_top_features(df, value_column, label, top_n=10):
    if df is None or df.empty:
        return f"### {label}\n\nNot available. The required package may not be installed or the step was skipped.\n"
    rows = []
    for _, row in df.head(top_n).iterrows():
        rows.append(f"- {row['feature']}: {row[value_column]:.6f}")
    return f"### {label}\n\n" + "\n".join(rows) + "\n"


def build_financial_interpretation(top_feature_names):
    discussed = [feature for feature in IMPORTANT_FEATURES if feature in top_feature_names]
    lines = [
        "Financial interpretation:",
        "",
        "- External source variables such as EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, and EXT_SOURCE_MEAN are important because they summarize outside risk signals. In this dataset, lower external-source scores usually indicate higher default risk.",
        "- Credit burden variables such as AMT_CREDIT, AMT_ANNUITY, CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, and CREDIT_GOODS_RATIO connect the requested loan size and required payment to the applicant's income.",
        "- Age and employment variables such as DAYS_BIRTH, AGE_YEARS, DAYS_EMPLOYED, and EMPLOYED_YEARS describe life-stage and job-stability patterns that can affect repayment risk.",
        "- Categorical variables such as NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, and OCCUPATION_TYPE describe employment, education, and work profile differences that may be related to financial stability.",
        "- AMT_REQ_CREDIT_BUREAU_YEAR can reflect recent credit inquiry behavior, which may be useful when evaluating credit-seeking pressure.",
        "",
    ]
    if discussed:
        lines.extend(
            [
                "Important requested features found among the top model signals:",
                "",
                ", ".join(discussed),
                "",
            ]
        )
    lines.extend(
        [
            "Overall reasonableness:",
            "",
            "The explanations should be considered reasonable if the strongest signals are dominated by external risk scores, repayment burden, income/annuity relationships, age or employment stability, and credit inquiry behavior. These are financially meaningful predictors. If a very high-ranked feature is hard to justify financially, it should be reviewed for possible data leakage or encoding artifacts.",
        ]
    )
    return "\n".join(lines)


def save_summary(xgb_importance_df, shap_importance_df, ebm_importance_df):
    top_names = set()
    for df in [xgb_importance_df, shap_importance_df, ebm_importance_df]:
        if df is not None and not df.empty:
            top_names.update(df.head(20)["feature"].astype(str).tolist())
            if "original_feature" in df.columns:
                top_names.update(df.head(20)["original_feature"].astype(str).tolist())

    summary = f"""# Experiment 3 Interpretability Summary

This summary explains the final XGBoost model and the EBM interpretable comparison model for Home Credit default-risk prediction. TARGET = 1 means payment difficulty / higher default risk.

{summarize_top_features(xgb_importance_df, "importance", "Top XGBoost Feature Importance")}

{summarize_top_features(shap_importance_df, "mean_abs_shap", "Top SHAP Importance")}

{summarize_top_features(ebm_importance_df, "importance", "Top EBM Feature Importance")}

{build_financial_interpretation(top_names)}
"""

    (TABLES_DIR / "experiment_3_interpretability_summary.md").write_text(
        summary,
        encoding="utf-8",
    )


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing Experiment 3 data...")
    data = prepare_experiment_data(quick_test=args.quick_test)
    print(f"Train shape: {data['X_train'].shape}")
    print(f"Validation shape: {data['X_valid'].shape}")
    print(f"Test shape: {data['X_test'].shape}")

    print("\n=== Training final XGBoost model ===")
    xgb_pipeline, xgb_params = build_xgboost_pipeline(data, quick_test=args.quick_test)
    print(f"XGBoost parameters: {xgb_params}")
    xgb_pipeline.fit(data["X_train"], data["y_train"])

    xgb_importance_df = save_xgboost_feature_importance(xgb_pipeline)
    print("Saved XGBoost feature importance outputs.")

    shap_importance_df = run_shap_analysis(
        xgb_pipeline,
        data,
        quick_test=args.quick_test,
    )
    if shap_importance_df is not None:
        print("Saved SHAP outputs.")

    ebm_importance_df = run_ebm_analysis(data, quick_test=args.quick_test)
    if ebm_importance_df is not None:
        print("Saved EBM outputs.")

    save_summary(xgb_importance_df, shap_importance_df, ebm_importance_df)
    print("\nExperiment 3 complete.")
    print(f"Summary: {TABLES_DIR / 'experiment_3_interpretability_summary.md'}")


if __name__ == "__main__":
    main()
