from typing import Any, Dict, Tuple

import pandas as pd
from pycaret.classification import setup, compare_models, pull


def compare_and_select_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    train_data = X_train.copy().reset_index(drop=True)
    train_data["target"] = y_train.values.ravel()

    val_data = X_val.copy().reset_index(drop=True)
    val_data["target"] = y_val.values.ravel()

    setup(
        data=train_data,
        target="target",
        test_data=val_data,
        index=False,
        preprocess=False,
        session_id=42,
        verbose=False,
    )

    best_model = compare_models(n_select=1, sort="AUC")
    comparison_df = pull()

    best_model_name = type(best_model).__name__
    best_model_params = best_model.get_params()
    best_row = comparison_df.iloc[0].to_dict()

    best_info = {
        "model_name": best_model_name,
        "parameters": {k: str(v) for k, v in best_model_params.items()},
        "metrics": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in best_row.items()},
    }

    return comparison_df, best_info, best_model
