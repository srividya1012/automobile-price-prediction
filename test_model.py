import pickle
import pandas as pd

model = pickle.load(open("models/ridge_model.pkl", "rb"))
poly = pickle.load(open("models/poly_transformer.pkl", "rb"))

# IMPORTANT: name must be NUMERIC
input_data = pd.DataFrame([{
    'name': 0,            
    'year': 2014,
    'km_driven': 120000,
    'fuel': 1,            
    'seller_type': 1,     
    'transmission': 1,    
    'owner': 1,           
    'mileage': 12.99,
    'engine': 2498,
    'max_power': 100.6,
    'seats': 8
}])

input_poly = poly.transform(input_data)
prediction = model.predict(input_poly)

print("Predicted Price:", prediction[0])
