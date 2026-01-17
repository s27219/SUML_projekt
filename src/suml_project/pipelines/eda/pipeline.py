from kedro.pipeline import Pipeline, node, pipeline
from .nodes import run_eda, generate_plots, save_analysis


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(run_eda, "raw_data", "eda_analysis", name="run_eda"),
            node(generate_plots, "raw_data", None, name="generate_plots"),
            node(save_analysis, "eda_analysis", None, name="save_analysis"),
        ]
    )
