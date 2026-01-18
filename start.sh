#!/bin/bash
uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000} &
streamlit run app/rain_prediction_ui.py --server.port ${PORT:-8501} --server.address 0.0.0.0