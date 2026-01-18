#!/usr/bin/env bash
set -e

STREAMLIT_PORT=8501
streamlit run app/rain_prediction_ui.py \
  --server.port ${STREAMLIT_PORT} \
  --server.address 127.0.0.1 \
  --server.headless true &

uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-10000}
