import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.main {
    background-color: transparent;
}
.hero-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    color: white;
    margin-bottom: 10px;
}
.hero-subtitle {
    text-align: center;
    color: #cfd8dc;
    font-size: 18px;
    margin-bottom: 40px;
}
.card {
    background-color: #1e1e1e;
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.4);
}
label {
    font-weight: 600 !important;
    color: #e0e0e0 !important;
}
.stButton>button {
    width: 100%;
    padding: 14px;
    font-size: 18px;
    border-radius: 12px;
    background: linear-gradient(90deg, #ff9800, #ff5722);
    color: white;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #ff5722, #ff9800);
}
.result-card {
    background: linear-gradient(135deg, #00c853, #2e7d32);
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    color: white;
    font-size: 26px;
    font-weight: 700;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)


model=pk.load(open('models/ridge_model.pkl','rb'))
poly = pk.load(open('models/poly_transformer.pkl', 'rb'))
st.header("Car Price Prediction")

cars_data = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    if pd.isna(car_name):
        return "Unknown"
    car_name = str(car_name)          # convert int/float to string
    car_name = car_name.split(' ')[0]
    return car_name.strip()
cars_data['name'] = cars_data['name'].apply(get_brand_name)

# ================= HERO SECTION =================
st.markdown("<div class='hero-title'>Car Price Prediction</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-subtitle'>Get instant resale value of your used car</div>",
    unsafe_allow_html=True
)

# ================= INPUT FORM =================
st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    name = st.selectbox("Car Brand", cars_data['name'].unique()) 
    year = st.number_input("Manufacturing Year", 1994, 2024)
    km_driven = st.number_input("Kilometers Driven", 11, 200000)
    fuel = st.selectbox("Fuel Type", cars_data['fuel'].unique())
    seller_type = st.selectbox("Seller Type", cars_data['seller_type'].unique())

with col2:
    transmission = st.selectbox("Transmission Type", cars_data['transmission'].unique())
    owner = st.selectbox("Owner Type", cars_data['owner'].unique())
    mileage = st.number_input("Mileage (km/l)", 10, 40)
    engine = st.number_input("Engine Capacity (CC)", 700, 5000)
    max_power = st.number_input("Max Power (bhp)", 0, 200)
    seats = st.number_input("Seats", 5, 10)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Predict Price"):
    input_data_model= pd.DataFrame({
        'name': [name],
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],     
        'owner': [owner],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats]})
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'],[1,2],inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[1,2,3],inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[1,2,3,4],inplace=True)
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'],[1,2,3,4,5],inplace=True)


    # input_data_model → DataFrame with 11 features

    input_data_poly = poly.transform(input_data_model)
    car_price = model.predict(input_data_poly)

    st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, #00c853, #2e7d32);
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        color: white;
        font-size: 26px;
        font-weight: 700;
        margin-top: 30px;
    ">
        Estimated Car Price<br>
        ₹ {round(car_price[0],2):,}
    </div>
    """,
    unsafe_allow_html=True
)
