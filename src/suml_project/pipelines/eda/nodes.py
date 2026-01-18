import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            unique = df[col].dropna().unique()
            if len(unique) == 2 and set(str(v).lower() for v in unique) <= {"yes", "no"}:
                df[col] = df[col].map({"Yes": 1, "No": 0, "yes": 1, "no": 0})
    return df


def run_eda(df: pd.DataFrame) -> Dict[str, Any]:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if "Date" in categorical_cols:
        categorical_cols.remove("Date")

    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100

    analysis = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing": {col: {"count": int(missing[col]), "percent": round(float(missing_pct[col]), 2)} for col in df.columns if missing[col] > 0},
        "numeric_stats": df[numeric_cols].describe().round(2).to_dict() if numeric_cols else {},
    }

    df_encoded = _encode_binary_columns(df)
    all_numeric = df_encoded.select_dtypes(include=[np.number]).columns.tolist()

    if len(all_numeric) >= 2:
        corr = df_encoded[all_numeric].corr()
        strong = []
        for i, c1 in enumerate(all_numeric):
            for c2 in all_numeric[i + 1:]:
                v = corr.loc[c1, c2]
                if pd.notna(v) and abs(v) > 0.5:
                    strong.append({"col1": c1, "col2": c2, "corr": round(float(v), 3)})
        strong.sort(key=lambda x: abs(x["corr"]), reverse=True)
        analysis["strong_correlations"] = strong

    if "RainTomorrow" in df.columns:
        vc = df["RainTomorrow"].value_counts()
        analysis["target_distribution"] = {str(k): int(v) for k, v in vc.items()}

    return analysis


def generate_plots(df: pd.DataFrame) -> None:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if "Date" in categorical_cols:
        categorical_cols.remove("Date")

    plots_dir = Path("data/reporting/eda/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / "numeric").mkdir(exist_ok=True)
    (plots_dir / "categorical").mkdir(exist_ok=True)

    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(s, bins=30, edgecolor="black")
        axes[0].set_title(f"Histogram: {col}")
        axes[1].boxplot(s, vert=False)
        axes[1].set_title(f"Boxplot: {col}")
        fig.tight_layout()
        fig.savefig(plots_dir / "numeric" / f"{col}.png", dpi=100)
        plt.close(fig)

    for col in categorical_cols:
        s = df[col].dropna().astype(str)
        if s.empty:
            continue
        vc = s.value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10, max(4, len(vc) * 0.4)))
        ax.barh(vc.index[::-1], vc.values[::-1], edgecolor="black")
        ax.set_title(f"Top {len(vc)}: {col}")
        fig.tight_layout()
        fig.savefig(plots_dir / "categorical" / f"{col}.png", dpi=100)
        plt.close(fig)

    df_encoded = _encode_binary_columns(df)
    all_numeric = df_encoded.select_dtypes(include=[np.number]).columns.tolist()

    if len(all_numeric) >= 2:
        corr = df_encoded[all_numeric].corr()
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(all_numeric)))
        ax.set_yticks(range(len(all_numeric)))
        ax.set_xticklabels(all_numeric, rotation=90)
        ax.set_yticklabels(all_numeric)
        ax.grid(False)
        for i in range(len(all_numeric)):
            for j in range(len(all_numeric)):
                val = corr.values[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
        ax.set_title("Correlation Matrix")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(plots_dir / "correlation_matrix.png", dpi=100)
        plt.close(fig)

    if "RainTomorrow" in df.columns and len(numeric_cols) >= 4:
        df_encoded = _encode_binary_columns(df)
        if "RainTomorrow" in df_encoded.columns:
            target_corr = df_encoded[numeric_cols].corrwith(df_encoded["RainTomorrow"]).abs().sort_values(ascending=False)
            top_features = target_corr.head(5).index.tolist()

            sample_df = df.sample(n=min(5000, len(df)), random_state=42).copy()

            plot_cols = top_features + ["RainTomorrow"]
            plot_data = sample_df[plot_cols].dropna()

            g = sns.pairplot(
                plot_data,
                hue="RainTomorrow",
                palette={"Yes": "#e74c3c", "No": "#3498db"},
                diag_kind="kde",
                corner=True,
                plot_kws={"alpha": 0.5, "s": 20},
            )
            g.fig.suptitle("Pairplot: Top 5 Features Correlated with RainTomorrow", y=1.02)
            g.savefig(plots_dir / "pairplot_top_features.png", dpi=100)
            plt.close(g.fig)


def save_analysis(analysis: Dict[str, Any]) -> None:
    output_dir = Path("data/reporting/eda")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eda_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
