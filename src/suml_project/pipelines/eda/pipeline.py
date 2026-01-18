from kedro.pipeline import Pipeline, node, pipeline
from .nodes import encode_binary_columns, run_eda, generate_plots, save_analysis


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(encode_binary_columns, "raw_data", "raw_data_encoded", name="encode_binary"),
            node(run_eda, ["raw_data", "raw_data_encoded"], "eda_analysis", name="run_eda"),
            node(generate_plots, ["raw_data", "raw_data_encoded"], None, name="generate_plots"),
            node(save_analysis, "eda_analysis", None, name="save_analysis"),
        ]
    )
