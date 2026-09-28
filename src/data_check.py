from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("data/raw/application_train.csv")
TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")


def load_data(data_path):
    """Load the training dataset from the raw data folder."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. "
            "Please place application_train.csv in data/raw/ first."
        )

    return pd.read_csv(data_path)


def print_basic_info(df):
    """Print dataset shape and all column names."""
    print("\n=== Dataset Shape ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n=== Column Names ===")
    for column in df.columns:
        print(column)


def print_target_distribution(df):
    """Print TARGET distribution as count and percentage."""
    if "TARGET" not in df.columns:
        raise ValueError("TARGET column was not found in the dataset.")

    target_counts = df["TARGET"].value_counts().sort_index()
    target_percentages = df["TARGET"].value_counts(normalize=True).sort_index() * 100

    target_summary = pd.DataFrame(
        {
            "count": target_counts,
            "percentage": target_percentages.round(2),
        }
    )

    print("\n=== TARGET Distribution ===")
    print(target_summary)

    return target_summary


def detect_column_types(df):
    """Detect numeric and categorical columns."""
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print("\n=== Column Type Counts ===")
    print(f"Numeric columns: {len(numeric_columns)}")
    print(f"Categorical columns: {len(categorical_columns)}")

    return numeric_columns, categorical_columns


def create_missing_value_summary(df):
    """Create a missing value summary table."""
    missing_summary = pd.DataFrame(
        {
            "column_name": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_percentage": (df.isna().mean().values * 100).round(2),
            "data_type": df.dtypes.astype(str).values,
        }
    )

    missing_summary = missing_summary.sort_values(
        by="missing_percentage",
        ascending=False,
    ).reset_index(drop=True)

    return missing_summary


def save_missing_value_outputs(missing_summary):
    """Save missing value summary tables."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    missing_values_path = TABLES_DIR / "missing_values.csv"
    top_missing_values_path = TABLES_DIR / "top_missing_values.csv"

    missing_summary.to_csv(missing_values_path, index=False)
    missing_summary.head(30).to_csv(top_missing_values_path, index=False)

    print("\n=== Saved Missing Value Tables ===")
    print(missing_values_path)
    print(top_missing_values_path)


def save_target_distribution_plot(target_summary):
    """Save a bar plot for TARGET class distribution."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_path = FIGURES_DIR / "class_distribution.png"

    plt.figure(figsize=(6, 4))
    bars = plt.bar(
        target_summary.index.astype(str),
        target_summary["count"],
        color=["#4C78A8", "#F58518"],
    )

    plt.title("TARGET Class Distribution")
    plt.xlabel("TARGET")
    plt.ylabel("Count")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\n=== Saved Class Distribution Plot ===")
    print(plot_path)


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print_basic_info(df)
    target_summary = print_target_distribution(df)
    detect_column_types(df)

    missing_summary = create_missing_value_summary(df)

    print("\n=== Top 30 Missing Value Columns ===")
    print(missing_summary.head(30))

    save_missing_value_outputs(missing_summary)
    save_target_distribution_plot(target_summary)

    print("\nData inspection finished.")


if __name__ == "__main__":
    main()
