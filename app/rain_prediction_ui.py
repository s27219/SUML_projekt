import streamlit as st
import requests
from datetime import date
import pandas as pd
from io import StringIO

st.set_page_config(
    page_title="Rain Tomorrow in Australia – Prediction Demo",
    page_icon="🌧️",
    layout="wide",
)

st.title("Rain Tomorrow in Australia")

API_URL = "http://localhost:8000/predict"
st.markdown("---")

prediction_method = st.radio(
    "Choose prediction method:",
    options=["Manual Input", "CSV Line"],
    horizontal=True
)

if prediction_method == "Manual Input":
    with st.container():
        st.subheader("Input today's weather conditions")

        col0, col1 = st.columns(2)

        with col0:
            location = st.selectbox(
                "Location",
                options=['','Adelaide', 'Albany', 'Albury', 'AliceSprings', 'BadgerysCreek', 'Ballarat', 'Bendigo',
                         'Brisbane', 'Cairns', 'Canberra', 'Cobar', 'CoffsHarbour', 'Dartmoor', 'Darwin', 'GoldCoast',
                         'Hobart', 'Katherine', 'Launceston', 'Melbourne', 'MelbourneAirport', 'Mildura', 'Moree',
                         'MountGambier', 'MountGinini', 'Newcastle', 'Nhil', 'NorahHead', 'NorfolkIsland', 'Nuriootpa',
                         'PearceRAAF', 'Penrith', 'Perth', 'PerthAirport', 'Portland', 'Richmond', 'Sale', 'SalmonGums',
                         'Sydney', 'SydneyAirport', 'Townsville', 'Tuggeranong', 'Uluru', 'WaggaWagga', 'Walpole',
                         'Watsonia', 'Williamtown', 'Witchcliffe', 'Wollongong', 'Woomera'],
            help="Australian city location"
            )

        with col1:
            selected_date = st.date_input(
                "Date",
                value=date.today(),
                help="Select the date for weather prediction"
            )

        col2, col3 = st.columns(2)

        with col2:
            min_temp = st.number_input(
                "Min Temperature (°C)",
                min_value=-10.0,
                max_value=50.0,
                value=15.0,
                step=0.5,
                help="Minimum temperature in degrees celsius"
            )

            max_temp = st.number_input(
                "Max Temperature (°C)",
                min_value=-10.0,
                max_value=50.0,
                value=25.0,
                step=0.5,
                help="Maximum temperature in degrees celsius"
            )

        with col3:
            temp_9am = st.number_input(
                "Temperature 9am (°C)",
                min_value=-10.0,
                max_value=50.0,
                value=18.0,
                step=0.5,
                help="Temperature at 9am in degrees celsius"
            )

            temp_3pm = st.number_input(
                "Temperature 3pm (°C)",
                min_value=-10.0,
                max_value=50.0,
                value=23.0,
                step=0.5,
                help="Temperature at 3pm in degrees celsius"
            )

        st.markdown("### Wind Conditions")
        col5, col6, col7 = st.columns(3)

        wind_directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

        with col5:
            wind_gust_dir = st.selectbox(
                "Wind Gust Direction",
                options=wind_directions,
                index=0,
                help="Direction of the strongest wind gust"
            )

            wind_gust_speed = st.number_input(
                "Wind Gust Speed (km/h)",
                min_value=0.0,
                max_value=150.0,
                value=40.0,
                step=1.0,
                help="Speed of the strongest wind gust in km/h"
            )

        with col6:
            wind_dir_9am = st.selectbox(
                "Wind Direction 9am",
                options=wind_directions,
                index=0,
                help="Wind direction at 9am"
            )

            wind_speed_9am = st.number_input(
                "Wind Speed 9am (km/h)",
                min_value=0.0,
                max_value=150.0,
                value=15.0,
                step=1.0,
                help="Wind speed at 9am in km/h"
            )

        with col7:
            wind_dir_3pm = st.selectbox(
                "Wind Direction 3pm",
                options=wind_directions,
                index=0,
                help="Wind direction at 3pm"
            )

            wind_speed_3pm = st.number_input(
                "Wind Speed 3pm (km/h)",
                min_value=0.0,
                max_value=150.0,
                value=20.0,
                step=1.0,
                help="Wind speed at 3pm in km/h"
            )

        st.markdown("### Humidity, Pressure and Rainfall")
        col8, col9, col10 = st.columns(3)

        with col8:
            humidity_3pm = st.number_input(
                "Humidity 3pm (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
                help="Relative humidity at 3pm"
            )

        with col9:
            pressure_3pm = st.number_input(
                "Pressure 3pm (hPa)",
                min_value=980.0,
                max_value=1050.0,
                value=1013.0,
                step=0.5,
                help="Atmospheric pressure at 3pm"
            )

        with col10:
            rainfall = st.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                max_value=500.0,
                value=0.0,
                step=0.1,
                help="The amount of rainfall recorded for the day in mm"
            )

    st.markdown("---")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        predict_clicked = st.button("Predict rain tomorrow")

    with col_right:
        if predict_clicked:
            if not location or location == '':
                st.error("Please select a location!")
            else:
                with st.spinner("Predicting..."):
                    try:
                        weather_data = {
                            "Location": location,
                            "MinTemp": min_temp,
                            "MaxTemp": max_temp,
                            "Rainfall": rainfall,
                            "WindGustDir": wind_gust_dir,
                            "WindGustSpeed": wind_gust_speed,
                            "WindDir9am": wind_dir_9am,
                            "WindDir3pm": wind_dir_3pm,
                            "WindSpeed9am": wind_speed_9am,
                            "WindSpeed3pm": wind_speed_3pm,
                            "Humidity3pm": humidity_3pm,
                            "Pressure3pm": pressure_3pm,
                            "Temp9am": temp_9am,
                            "Temp3pm": temp_3pm,
                            "Date": selected_date.strftime("%Y-%m-%d")
                        }

                        response = requests.post(API_URL, json=weather_data)

                        if response.status_code == 200:
                            result = response.json()
                            probability = result["probability"]

                            st.metric(
                                label="Predicted probability of rain tomorrow",
                                value=f"{probability:.1f}%",
                            )

                            if probability > 50:
                                st.success("🌧️ Rain is likely tomorrow!")
                            else:
                                st.info("☀️ Rain is unlikely tomorrow.")
                        else:
                            st.error(f"Error: {response.status_code}")

                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to the API. Make sure FastAPI server is running.")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
        else:
            st.caption("Fill in the inputs and click **Predict rain tomorrow** to see the model output.")

else:
    st.subheader("Paste CSV line")

    st.info("""
            Paste a CSV line with the following format (comma-separated):
            
            `Location,MinTemp,MaxTemp,Rainfall,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity3pm,Pressure3pm,Temp9am,Temp3pm,Date`
            
            Example:
            `Sydney,15.5,25.3,0.0,NW,40,N,NE,15,20,50,1013.0,18.0,23.0,2024-01-15`
            """)

    csv_line = st.text_area(
        "CSV Line",
        height=100,
        placeholder="Sydney,15.5,25.3,0.0,NW,40,N,NE,15,20,50,1013.0,18.0,23.0,2024-01-15"
    )

    st.markdown("---")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        predict_from_csv = st.button("Predict from CSV")

    with col_right:
        if predict_from_csv:
            if not csv_line.strip():
                st.error("Please paste a CSV line first!")
            else:
                with st.spinner("Predicting..."):
                    try:
                        header = "Location,MinTemp,MaxTemp,Rainfall,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity3pm,Pressure3pm,Temp9am,Temp3pm,Date"
                        csv_data = f"{header}\n{csv_line.strip()}"

                        df = pd.read_csv(StringIO(csv_data))

                        if len(df) == 0:
                            st.error("Could not parse CSV line!")
                        else:
                            row = df.iloc[0]

                            weather_data = {
                                "Location": str(row["Location"]),
                                "MinTemp": float(row["MinTemp"]),
                                "MaxTemp": float(row["MaxTemp"]),
                                "Rainfall": float(row["Rainfall"]),
                                "WindGustDir": str(row["WindGustDir"]),
                                "WindGustSpeed": float(row["WindGustSpeed"]),
                                "WindDir9am": str(row["WindDir9am"]),
                                "WindDir3pm": str(row["WindDir3pm"]),
                                "WindSpeed9am": float(row["WindSpeed9am"]),
                                "WindSpeed3pm": float(row["WindSpeed3pm"]),
                                "Humidity3pm": float(row["Humidity3pm"]),
                                "Pressure3pm": float(row["Pressure3pm"]),
                                "Temp9am": float(row["Temp9am"]),
                                "Temp3pm": float(row["Temp3pm"]),
                                "Date": str(row["Date"])
                            }

                            response = requests.post(API_URL, json=weather_data)

                            if response.status_code == 200:
                                result = response.json()
                                probability = result["probability"]

                                st.metric(
                                    label="Predicted probability of rain tomorrow",
                                    value=f"{probability:.1f}%",
                                )

                                if probability > 50:
                                    st.success("🌧️ Rain is likely tomorrow!")
                                else:
                                    st.info("☀️ Rain is unlikely tomorrow.")
                            else:
                                st.error(f"Error: {response.status_code}")

                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to the API. Make sure FastAPI server is running.")
                    except pd.errors.ParserError:
                        st.error("Invalid CSV format! Make sure values are comma-separated.")
                    except KeyError as e:
                        st.error(f"Missing required column: {str(e)}")
                    except ValueError as e:
                        st.error(f"Invalid value in CSV: {str(e)}")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
        else:
            st.caption("Paste a CSV line and click **Predict from CSV** to see the model output.")
