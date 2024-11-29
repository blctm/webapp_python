import streamlit as st
import pandas as pd
import joblib
import os

# Get the full path to the current directory
current_dir = os.path.dirname(os.path.abspath("scaler.pkl"))
print(current_dir)

# Load the scaler and model with full paths
scaler_path = r"C:\Users\blinares\MINETHIC\scaler.pkl"
model_path = r"C:\Users\blinares\MINETHIC\random_forest_model.pkl"

# Load the scaler and model
scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

# App title
st.title("BS Eficiencia Predictor")
# Sidebar inputs for numerical features
st.sidebar.header("Input Features")
MP_gr = st.sidebar.number_input("MP(gr)", value=0.0)
Residuo_gr = st.sidebar.number_input("Residuo (gr)", value=0.0)
MP = st.sidebar.number_input("MP", value=0.0)  # Added this line
Ph = st.sidebar.number_input("Ph", value=0.0)
Tiempo = st.sidebar.number_input("Tiempo", value=0)
Temperatura = st.sidebar.number_input("Temperatura", value=0)
LixivLiqui = st.sidebar.number_input("LixivLiqui", value=0.0)
LicorLavado = st.sidebar.number_input("LicorLavado", value=0.0)
# Dropdown for Materia Prima
materia_prima = st.sidebar.selectbox(
    "Materia Prima",
    options=["Al", "Fe", "Mg", "Mn", "Zn"],
    index=0
)

# Convert Materia Prima to one-hot encoded features
materia_prima_dict = {
    "Fe": [1, 0, 0, 0],
    "Mg": [0, 1, 0, 0],
    "Mn": [0, 0, 1, 0],
    "Zn": [0, 0, 0, 1],
    "Al": [0, 0, 0, 0]  # Al corresponds to all False
}
materia_prima_features = materia_prima_dict[materia_prima]
# Combine inputs into a single DataFrame
input_data = pd.DataFrame([{
    "MP(gr)": MP_gr,
    "Residuo (gr)": Residuo_gr,
    "MP": MP,
    "Ph": Ph,
    "Tiempo": Tiempo,
    "Temperatura": Temperatura,
    "LixivLiqui": LixivLiqui,
    "LicorLavado": LicorLavado,
    "Materia Prima_Fe": materia_prima_features[0],
    "Materia Prima_Mg": materia_prima_features[1],
    "Materia Prima_Mn": materia_prima_features[2],
    "Materia Prima_Zn": materia_prima_features[3],
}])
# Scale the input data
scaled_data = scaler.transform(input_data)
# Scale the input data
scaled_data = scaler.transform(input_data)

# Make predictions when the "Predict" button is clicked
if st.button("Predict"):
    prediction = model.predict(scaled_data)
    st.write(f"### Predicted BS Eficiencia: {prediction[0]}")