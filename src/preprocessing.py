from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/raw/application_train.csv")
RANDOM_STATE = 42
VALID_SIZE = 0.15
TEST_SIZE = 0.15


def load_data(data_path=DATA_PATH):
    """Load application_train.csv from the raw data folder."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. "
            "Please place application_train.csv in data/raw/ first."
        )

    return pd.read_csv(data_path)


def safe_divide(numerator, denominator):
    """Divide safely. Zero or missing denominator becomes NaN."""
    safe_denominator = denominator.where(denominator.notna() & (denominator != 0), np.nan)
    result = numerator / safe_denominator
    return result.replace([np.inf, -np.inf], np.nan)


def add_financial_features(df):
    """Add simple, interpretable financial features."""
    df = df.copy()

    df["CREDIT_INCOME_RATIO"] = safe_divide(
        df["AMT_CREDIT"],
        df["AMT_INCOME_TOTAL"],
    )
    df["ANNUITY_INCOME_RATIO"] = safe_divide(
        df["AMT_ANNUITY"],
        df["AMT_INCOME_TOTAL"],
    )
    df["CREDIT_GOODS_RATIO"] = safe_divide(
        df["AMT_CREDIT"],
        df["AMT_GOODS_PRICE"],
    )

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365

    # In Home Credit, DAYS_EMPLOYED can contain abnormal positive values such as 365243.
    valid_days_employed = df["DAYS_EMPLOYED"].where(df["DAYS_EMPLOYED"] <= 0, np.nan)
    df["EMPLOYED_YEARS"] = -valid_days_employed / 365

    ext_source_features = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN"] = df[ext_source_features].mean(axis=1)
    df["EXT_SOURCE_STD"] = df[ext_source_features].std(axis=1)

    return df


def split_features_target(df):
    """Use TARGET as y and all other columns except SK_ID_CURR as X."""
    if "TARGET" not in df.columns:
        raise ValueError("TARGET column was not found in the dataset.")

    y = df["TARGET"]
    X = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")

    return X, y


def split_train_valid_test(X, y):
    """Create stratified 70% train, 15% validation, and 15% test splits."""
    valid_test_size = VALID_SIZE + TEST_SIZE

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=valid_test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    test_share_from_temp = TEST_SIZE / valid_test_size

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_share_from_temp,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def get_feature_types(X):
    """Detect numeric and categorical features after feature engineering."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return numeric_features, categorical_features


def make_one_hot_encoder():
    """Create OneHotEncoder for both newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_linear_preprocessor(numeric_features, categorical_features):
    """Build preprocessing for Logistic Regression."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("one_hot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_tree_preprocessor(numeric_features, categorical_features):
    """Build preprocessing for tree-based models and EBM."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("one_hot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def prepare_data():
    """Prepare data and preprocessors for Experiment 1."""
    df = load_data()
    df = add_financial_features(df)
    X, y = split_features_target(df)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_train_valid_test(X, y)

    numeric_features, categorical_features = get_feature_types(X_train)
    linear_preprocessor = build_linear_preprocessor(
        numeric_features,
        categorical_features,
    )
    tree_preprocessor = build_tree_preprocessor(
        numeric_features,
        categorical_features,
    )

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "linear_preprocessor": linear_preprocessor,
        "tree_preprocessor": tree_preprocessor,
    }


def print_summary(prepared_data):
    """Print a simple summary when this file is run directly."""
    print("\n=== Split Shapes ===")
    print(f"X_train: {prepared_data['X_train'].shape}")
    print(f"X_valid: {prepared_data['X_valid'].shape}")
    print(f"X_test: {prepared_data['X_test'].shape}")

    print("\n=== TARGET Distribution By Split ===")
    for split_name in ["y_train", "y_valid", "y_test"]:
        y_split = prepared_data[split_name]
        distribution = pd.DataFrame(
            {
                "count": y_split.value_counts().sort_index(),
                "percentage": (
                    y_split.value_counts(normalize=True).sort_index() * 100
                ).round(2),
            }
        )
        print(f"\n{split_name}")
        print(distribution)

    print("\n=== Feature Column Counts ===")
    print(f"Numeric features: {len(prepared_data['numeric_features'])}")
    print(f"Categorical features: {len(prepared_data['categorical_features'])}")

    print("\nPreprocessors created:")
    print("- linear_preprocessor: median imputation + StandardScaler + one-hot encoding")
    print("- tree_preprocessor: median imputation + one-hot encoding")


def main():
    prepared_data = prepare_data()
    print_summary(prepared_data)


if __name__ == "__main__":
    main()
