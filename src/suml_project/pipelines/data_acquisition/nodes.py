from pathlib import Path

import kagglehub
import pandas as pd


def acquire_data() -> pd.DataFrame:
    path = kagglehub.dataset_download("jsphyg/weather-dataset-rattle-package")

    path = Path(path)

    csv_path = path / "weatherAUS.csv"

    df = pd.read_csv(csv_path)
    return df