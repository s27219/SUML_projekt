import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        strong = []
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1:]:
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

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=90)
        ax.set_yticklabels(numeric_cols)
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                val = corr.values[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
        ax.set_title("Correlation Matrix")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(plots_dir / "correlation_matrix.png", dpi=100)
        plt.close(fig)


def save_analysis(analysis: Dict[str, Any]) -> None:
    output_dir = Path("data/reporting/eda")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eda_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
