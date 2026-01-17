from kedro.pipeline import Pipeline, node
from .nodes import compare_and_select_models


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=compare_and_select_models,
                inputs=["X_train", "y_train", "X_val", "y_val"],
                outputs=["model_comparison", "best_model_info", "best_model"],
                name="compare_and_select_models",
            ),
        ]
    )
