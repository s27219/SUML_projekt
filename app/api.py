from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

app = FastAPI()

model = joblib.load(MODELS_DIR / "production_model.pkl")
encoders = joblib.load(MODELS_DIR / "label_encoders.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")

class WeatherData(BaseModel):
    Location: str
    MinTemp: float
    MaxTemp: float
    Rainfall: float
    WindGustDir: str
    WindGustSpeed: float
    WindDir9am: str
    WindDir3pm: str
    WindSpeed9am: float
    WindSpeed3pm: float
    Humidity3pm: float
    Pressure3pm: float
    Temp9am: float
    Temp3pm: float
    Date: str

def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" not in df.columns:
        return df
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["Date"].dt.dayofyear / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["Date"].dt.dayofyear / 365)
    df = df.drop(columns=["Date"])
    return df

def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df = extract_date_features(df)
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        if col in encoders:
            try:
                df[col] = encoders[col].transform(df[col].astype(str))
            except ValueError:
                df[col] = 0
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cols_to_scale = [col for col in numeric_cols if col in scaler.feature_names_in_]
    if cols_to_scale:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df

@app.post("/predict")
async def predict_rain(data: WeatherData):
    try:
        input_dict = data.dict()
        input_df = preprocess_input(input_dict)
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)
        return {
            "prediction": int(predictions[0]),
            "probability": float(probabilities[0][1] * 100)
        }
    except Exception as e:
        return {
            "error": str(e),
            "prediction": 0,
            "probability": 0.0
        }

@app.get("/health")
async def health_check():
    return {"status": "ok"}
