from kedro.pipeline import Pipeline, node
from suml_project.pipelines.data_acquisition.nodes import acquire_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=acquire_data,
            inputs=None,
            outputs="raw_data",
            name="acquire_data_node",
        )
    ])
