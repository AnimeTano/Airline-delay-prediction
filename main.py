import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('models/xgb_model.pk1')
encoder = joblib.load('models/target_encoder.pk1')
scaler = joblib.load('models/scaler.pk1')

X_train = pd.read_csv('data/processed/X_train.csv')
feature_columns = X_train.columns.tolist()
lag_cols = [col for col in feature_columns if 'lag' in col]
avg_lags = X_train[lag_cols].mean()

st.title("Airline Delay Prediction")
st.write("Предсказание проблемного месяца для авиакомпаний по аэропортам США")

col1, col2 = st.columns(2)
with col1:
    carrier = st.selectbox("Авиакомпания", ['AA', 'DL', 'UA', 'WN', 'B6', 'NK', 'AS', 'F9', 'HA', 'OO', 'MQ', 'YX'])
    airport = st.selectbox("Аэропорт", ['ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'JFK', 'SFO', 'SEA', 'MIA', 'IAH'])
    month = st.slider('Месяц', 1, 12, 6)

with col2:
    arr_flights = st.number_input('Количество рейсов', min_value=1, value=100)
    arr_cancelled = st.number_input('Отменённые рейсы', min_value=0, value=5)
    arr_diverted = st.number_input('Отклонённые рейсы', min_value=0, value=2)

input_df = pd.DataFrame(columns=feature_columns)
input_df.loc[0] = 0

input_df.loc[0, 'carrier'] = carrier
input_df.loc[0, 'airport'] = airport
input_df.loc[0, 'month'] = month
input_df.loc[0, 'arr_flights'] = arr_flights
input_df.loc[0, 'arr_cancelled'] = arr_cancelled
input_df.loc[0, 'arr_diverted'] = arr_diverted

input_df.loc[0, 'month_sin'] = np.sin(2 * np.pi * month / 12)
input_df.loc[0, 'month_cos'] = np.cos(2 * np.pi * month / 12)

def get_season(m):
    if m in [12, 1, 2]: return 'winter'
    elif m in [3, 4, 5]: return 'spring'
    elif m in [6, 7, 8]: return 'summer'
    else: return 'autumn'
input_df.loc[0, 'season'] = get_season(month)

# Заполняем лаги средними
for col in lag_cols:
    input_df.loc[0, col] = avg_lags[col]

if st.button("Predict"):
    input_enc = encoder.transform(input_df)
    input_scaled = scaler.transform(input_enc)
    prob = model.predict_proba(input_scaled)[0, 1]

    st.metric("Вероятность проблемного месяца", f"{prob:.2%}")
    if prob > 0.5:
        st.warning("Высокий риск проблемного месяца")
    else:
        st.success("Низкий риск")