from kedro.pipeline import Pipeline, node
from .nodes import train_final_model, evaluate_on_test


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_final_model,
                inputs=["best_model", "X_train", "y_train", "X_val", "y_val"],
                outputs="production_model",
                name="train_final_model",
            ),
            node(
                func=evaluate_on_test,
                inputs=["production_model", "X_test", "y_test"],
                outputs=["test_metrics", "test_predictions", "confusion_matrix"],
                name="evaluate_on_test",
            ),
        ]
    )
