from kedro.pipeline import Pipeline, node
from .nodes import compare_and_select_models


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=compare_and_select_models,
                inputs=["X_train", "y_train", "X_test", "y_test"],
                outputs="model_selection_result",
                name="compare_and_select_models",
            ),
        ]
    )
