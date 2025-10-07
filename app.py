# -------------------------------
# 1️⃣ Import Libraries
# -------------------------------
import streamlit as st
import pandas as pd
from math import exp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# 2️⃣ Load datasets
# -------------------------------
dataset1 = pd.read_csv("Crop_recommendation.csv")
dataset3 = pd.read_csv("crop_and_soil.csv")

# -------------------------------
# 3️⃣ Clean column names
# -------------------------------
def clean_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

dataset1 = clean_columns(dataset1)
dataset3 = clean_columns(dataset3)

# -------------------------------
# 4️⃣ Standardize important columns
# -------------------------------
dataset1.rename(columns={
    'label': 'crop',
    'temperature': 'temp_c',
    'humidity': 'humidity_percent',
    'rainfall': 'rainfall_mm'
}, inplace=True)

dataset3.rename(columns={
    'crop_type':'crop',
    'soil_type':'soil_type',
    'temperature':'temp_c',
    'humidity':'humidity_percent',
    'moisture':'soil_moisture_percent',
    'nitrogen':'n',
    'potassium':'k',
    'phosphorous':'p'
}, inplace=True)

# -------------------------------
# 5️⃣ Merge datasets safely
# -------------------------------
merged = pd.merge(dataset1, dataset3, on='crop', how='left')
merged['soil_type'] = merged.get('soil_type', pd.Series('Unknown')).fillna('Unknown')
merged['soil_moisture_percent'] = merged.get('soil_moisture_percent', pd.Series(40)).fillna(40)
for col in ['n','p','k','temp_c','humidity_percent']:
    if col not in merged.columns:
        merged[col] = 50

# -------------------------------
# 6️⃣ Define crop factor multipliers
# -------------------------------
crop_factors = {}
for crop_name in merged['crop'].unique():
    crop_factors[crop_name] = 0.8 + 0.4 * hash(crop_name) % 100 / 100

# -------------------------------
# 7️⃣ FAO-56 Penman-Monteith formula for CWR
# -------------------------------
def calculate_cwr(Tmax, Tmin, RH_mean, Rs=15, u2=2, z=50):
    Tmean = (Tmax + Tmin) / 2
    es_Tmax = 0.6108 * exp((17.27*Tmax)/(Tmax+237.3))
    es_Tmin = 0.6108 * exp((17.27*Tmin)/(Tmin+237.3))
    es = (es_Tmax + es_Tmin)/2
    ea = es * (RH_mean/100)
    delta = 4098 * (0.6108 * exp((17.27*Tmean)/(Tmean+237.3))) / ((Tmean+237.3)**2)
    P = 101.3 * (((293 - 0.0065*z)/293)**5.26)
    gamma = 0.000665 * P
    Rn = 0.77 * Rs
    G = 0
    ET0 = (0.408 * delta * (Rn - G) + gamma * (900/(Tmean + 273)) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34*u2))
    return ET0

# -------------------------------
# 8️⃣ Prepare irrigation ML model
# -------------------------------
merged['CWR_mm_per_day'] = merged.apply(lambda row: calculate_cwr(
    Tmax=row['temp_c']+5,
    Tmin=row['temp_c']-5,
    RH_mean=row['humidity_percent']
) * crop_factors[row['crop']], axis=1)

def irrigation_label(row):
    if row['CWR_mm_per_day'] > 0 and row['soil_moisture_percent'] < 10:
        return 'Drip/Furrow'
    elif row['CWR_mm_per_day'] > 10 and row['soil_moisture_percent'] < 20:
        return 'Drip/Sprinkler'
    elif row['CWR_mm_per_day'] > 20 and row['soil_moisture_percent'] < 30:
        return 'Sprinkler/Surface'
    else:
        return 'Light Surface/Supplemental Sprinkler'

merged['irrigation_type'] = merged.apply(irrigation_label, axis=1)

crop_encoder = LabelEncoder()
merged['crop_encoded'] = crop_encoder.fit_transform(merged['crop'])

features_clf = ['crop_encoded','CWR_mm_per_day','n','p','k','soil_moisture_percent']
X_clf = merged[features_clf]
y_clf = merged['irrigation_type']

clf_model = RandomForestClassifier(n_estimators=200, random_state=42)
clf_model.fit(X_clf, y_clf)
joblib.dump(clf_model, 'irrigation_model.pkl')

# -------------------------------
# 9️⃣ Streamlit Interface
# -------------------------------
st.title("ML-based Crop Water Requirement & Irrigation Prediction")

st.sidebar.header("Input Crop Data")
crop_list = merged['crop'].dropna().unique().tolist()
crop = st.sidebar.selectbox("Select Crop", crop_list)
temp = st.sidebar.number_input("Temperature (°C)", 10.0, 45.0, 25.0)
humidity = st.sidebar.number_input("Humidity (%)", 10.0, 100.0, 70.0)
N = st.sidebar.number_input("Nitrogen (N)", 0, 100, 50)
P = st.sidebar.number_input("Phosphorous (P)", 0, 100, 50)
K = st.sidebar.number_input("Potassium (K)", 0, 100, 50)
soil_moisture = st.sidebar.number_input("Soil Moisture (%)", 0, 100, 40)

if st.button("Predict"):
    factor = crop_factors[crop]
    Tmax = temp + 5
    Tmin = temp - 5
    cwr_pred = calculate_cwr(Tmax, Tmin, humidity) * factor

    crop_enc = crop_encoder.transform([crop])[0]
    clf_model = joblib.load('irrigation_model.pkl')
    irrigation_pred = clf_model.predict([[crop_enc, cwr_pred, N, P, K, soil_moisture]])[0]

    st.write(f"**Selected Crop:** {crop}")
    st.write(f"**Predicted Crop Water Requirement (CWR):** {cwr_pred:.2f} mm/day")
    st.write(f"**Recommended Irrigation Type:** {irrigation_pred}")

    # -------------------------------
    # 1️⃣0️⃣ Plot CWR vs Temperature for this crop
    # -------------------------------
    temps = np.arange(10, 41, 1)
    cwr_values = [calculate_cwr(t+5, t-5, humidity) * factor for t in temps]

    fig, ax = plt.subplots()
    ax.plot(temps, cwr_values, marker='o', color='green')
    ax.set_title(f"CWR vs Temperature for {crop}")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Crop Water Requirement (mm/day)")
    ax.grid(True)
    st.pyplot(fig)
