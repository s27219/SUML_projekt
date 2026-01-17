from kedro.pipeline import Pipeline, node

from suml_project.pipelines.preprocessing.nodes import (
    create_final_datasets,
    drop_high_missing_columns,
    drop_target_nulls,
    encode_categorical_features,
    impute_missing_values,
    scale_features,
    split_features_target,
    split_train_val_test,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=drop_high_missing_columns,
                inputs=["raw_data", "params:missing_threshold"],
                outputs="data_reduced_columns",
                name="drop_high_missing_columns_node",
            ),
            node(
                func=drop_target_nulls,
                inputs=["data_reduced_columns", "params:target_column"],
                outputs="data_no_target_nulls",
                name="drop_target_nulls_node",
            ),
            node(
                func=impute_missing_values,
                inputs="data_no_target_nulls",
                outputs="data_imputed",
                name="impute_missing_values_node",
            ),
            node(
                func=encode_categorical_features,
                inputs=["data_imputed", "params:target_column"],
                outputs=["data_encoded", "label_encoders"],
                name="encode_categorical_features_node",
            ),
            node(
                func=split_features_target,
                inputs=["data_encoded", "params:target_column"],
                outputs=["features", "target"],
                name="split_features_target_node",
            ),
            node(
                func=split_train_val_test,
                inputs=[
                    "features",
                    "target",
                    "params:test_size",
                    "params:val_size",
                    "params:random_state",
                ],
                outputs="data_splits",
                name="split_train_val_test_node",
            ),
            node(
                func=scale_features,
                inputs=["data_splits", "params:numeric_columns"],
                outputs=["scaled_splits", "scaler"],
                name="scale_features_node",
            ),
            node(
                func=create_final_datasets,
                inputs="scaled_splits",
                outputs=[
                    "X_train",
                    "X_val",
                    "X_test",
                    "y_train",
                    "y_val",
                    "y_test",
                ],
                name="create_final_datasets_node",
            ),
        ]
    )