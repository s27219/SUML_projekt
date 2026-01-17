from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def drop_high_missing_columns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    return df.drop(columns=cols_to_drop)


def drop_target_nulls(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    return df.dropna(subset=[target_column])


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_categorical_features(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    df = df.copy()
    encoders = {}

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if "Date" in df.columns:
        df = df.drop(columns=["Date"])
        if "Date" in categorical_cols:
            categorical_cols.remove("Date")

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    random_state: int,
) -> dict[str, Any]:
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def scale_features(
    splits: dict[str, Any],
    numeric_columns: list[str],
) -> tuple[dict[str, Any], StandardScaler]:
    scaler = StandardScaler()

    X_train = splits["X_train"].copy()
    X_val = splits["X_val"].copy()
    X_test = splits["X_test"].copy()

    cols_to_scale = [col for col in numeric_columns if col in X_train.columns]

    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    scaled_splits = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": splits["y_train"],
        "y_val": splits["y_val"],
        "y_test": splits["y_test"],
    }

    return scaled_splits, scaler


def create_final_datasets(
    scaled_splits: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        scaled_splits["X_train"],
        scaled_splits["X_val"],
        scaled_splits["X_test"],
        scaled_splits["y_train"],
        scaled_splits["y_val"],
        scaled_splits["y_test"],
    )


def balance_by_feature(
    df: pd.DataFrame,
    feature_column: str,
    target_ratio: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    import numpy as np

    df = df.copy()

    class_counts = df[feature_column].value_counts()
    minority_class = class_counts.idxmin()
    minority_count = class_counts[minority_class]

    if target_ratio is None:
        target_majority_count = minority_count
    else:
        target_majority_count = int(minority_count / target_ratio)

    minority_indices = df[df[feature_column] == minority_class].index
    majority_indices = df[df[feature_column] != minority_class].index

    np.random.seed(random_state)
    sampled_majority_indices = np.random.choice(
        majority_indices,
        size=min(target_majority_count, len(majority_indices)),
        replace=False
    )

    balanced_indices = list(minority_indices) + list(sampled_majority_indices)
    np.random.shuffle(balanced_indices)

    return df.loc[balanced_indices]


def remove_outliers(
    df: pd.DataFrame,
    target_column: str | None = None,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    if target_column and target_column in numeric_cols:
        numeric_cols.remove(target_column)

    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR

    mask = ~((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)

    df_cleaned = df[mask]

    print(f"Removed {len(df) - len(df_cleaned)} outlier rows ({((len(df) - len(df_cleaned)) / len(df) * 100):.2f}%)")
    print(f"Remaining rows: {len(df_cleaned)}")

    return df_cleaned