import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from pycaret.classification import setup, compare_models, pull, save_model, get_config


def compare_and_select_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Dict[str, Any]:
    train_data = X_train.copy().reset_index(drop=True)
    train_data["target"] = y_train.values.ravel()

    test_data = X_test.copy().reset_index(drop=True)
    test_data["target"] = y_test.values.ravel()

    setup(
        data=train_data,
        target="target",
        test_data=test_data,
        index=False,
        preprocess=False,
        session_id=42,
        verbose=False,
    )

    best_model = compare_models(n_select=1, sort="AUC")
    comparison_df = pull()

    output_dir = Path("data/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(output_dir / "model_comparison.csv", index=True)

    save_model(best_model, str(output_dir / "best_model"))

    best_model_name = type(best_model).__name__
    best_model_params = best_model.get_params()
    best_row = comparison_df.iloc[0].to_dict()

    best_info = {
        "model_name": best_model_name,
        "parameters": {k: str(v) for k, v in best_model_params.items()},
        "metrics": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in best_row.items()},
    }

    with open(output_dir / "best_model_info.json", "w") as f:
        json.dump(best_info, f, indent=2)

    return {"best_model_name": best_model_name, "comparison_rows": len(comparison_df)}
