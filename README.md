# SUML_project

## Instrukcje uruchomienia projektu

Zainstaluj wymagane biblioteki
```
pip install -r requirements.txt
```

Uruchom wszystkie pipeline'y
```
kedro run
```

Uruchom jeden z pipeline'ów
```
kedro run --pipeline <nazwa-pipeline'u>
```

Uruchom aplikację backend
```
uvicorn app.api:app --reload
```

Uruchom aplikację frontend
```
streamlit run app/rain_prediction_ui.py
```