#!/usr/bin/env bash
set -e

uvicorn app.api:app --host 127.0.0.1 --port 8000 &

streamlit run app/rain_prediction_ui.py \
  --server.port ${PORT:-10000} \
  --server.address 0.0.0.0 \
  --server.headless true
