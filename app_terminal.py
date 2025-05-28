import streamlit as st
import os
from joblib import load
import pandas as pd
import sklearn
import numpy as np


SUBURBS = [ "Auckland", "Avondale", "Blockhouse Bay", "Eden Terrace", "Ellerslie", 
        "Epsom", "Freemans Bay", "Glen Innes", "Glendowie", "Grafton", "Greenlane", 
        "Grey Lynn", "Herne Bay", "Hillsborough", "Kingsland", "Kohimarama", 
        "Meadowbank", "Mission Bay", "Morningside", "Mount Albert", "Mount Eden", 
        "Mount Roskill", "Mount Wellington", "New Windsor", "Newmarket", "One Tree Hill", 
        "Onehunga", "Otahuhu", "Parnell", "Point Chevalier", "Point England", 
        "Ponsonby", "Remuera", "Royal Oak", "Saint Heliers", "Saint Johns", 
        "Saint Marys Bay", "Sandringham", "St Heliers", "Stonefields", "Three Kings", 
        "Waiotaiki Bay", "Waterview", "Westmere"
    ]

st.warning("Modelo não carregado. Treine e carregue um modelo válido para realizar previsões.")
scaler_features = load('scaler_features.joblib')
target_features = load('scaler_target.joblib')

def predict_price(data):

    prediction_data = {
        "Suburb": 0,
        "Bedroom": 0,
        "Bathroom": 0,
        "Garage": 0,
        "Furniture": 0,
        "Type_Apartment": 0,
        "Type_House": 0,
        "Type_Studio": 0,
        "Type_Townhouse": 0,
        "Type_Unit": 0,
    }

    # create and array populate the type with iether 0 or 1
    property_type = ["Apartment", "House", "Studio", "Townhouse", "Unit"]
    col_name = ["Type_Apartment","Type_House","Type_Studio","Type_Townhouse","Type_Unit"]

    property_type_bin = [0,0,0,0,0]
    for i in range(len(property_type)):
        if data['property_type'] == property_type[i]:
            property_type_bin[i] = 1
        prediction_data[col_name[i]] = property_type_bin[i]

    #find suburb index, matches to how we encode suburbs in the model.
    for i in range(len(SUBURBS)):
        if data['suburb'] == SUBURBS[i]:
            prediction_data['Suburb'] = i

    prediction_data['Bedroom'] = data['num_bedrooms']
    prediction_data['Bathroom'] = data['num_bathrooms']
    prediction_data['Garage'] = data['num_garage_spaces']
    prediction_data['Furniture'] = data['is_furnished']
    
    #normalize the input data using the scaler from training
    prediction_data = pd.DataFrame(prediction_data, index=[0])
    normalized_data_array = scaler_features.transform(prediction_data)
    normalized_data = pd.DataFrame(normalized_data_array, columns=prediction_data.columns)
    
    # make prediction ond un-normalize the data
    prediction = model.predict(normalized_data)
    prediction = prediction.reshape(-1, 1)
    prediction = target_features.inverse_transform(prediction)
    prediction = prediction.flatten()
    
    return prediction[0]

def main():
    st.title("Rent Price Predictor App")
    st.write("### Property Details")

    # Select suburb from the list
    suburb = st.selectbox("Select the suburb:", [""] + SUBURBS)

    # Select property type
    property_type = st.selectbox("Select property type:", ["", "House", "Townhouse", "Unit", "Apartment","Studio", "Unknown"])

    # Number of rooms
    # num_rooms = st.slider("How many rooms?", 1, 10)

    # Number of bedrooms
    num_bedrooms = st.slider("How many bedrooms?", 1, 10)
    num_bathrooms = st.slider("How many bathrooms?", 1, 5)

    # Number of garage spaces
    num_garage_spaces = st.slider("How many garage spaces?", 0, 10)

    # Furnishing details
    is_furnished = st.radio("Is the property furnished?", ["Yes", "No"])
    is_furnished_binary = 1 if is_furnished == "Yes" else 0

    # Property price
    price_str = st.text_input("Enter the price:")
    try:
        price = float(price_str)
    except ValueError:
        st.write("Please enter a valid number.")
        price = None

    predicted_price = 0
    if st.button("Predict") and price:
        data = {
            "suburb": suburb,
            "property_type": property_type,
            # "num_rooms": num_rooms,
            "num_bedrooms": num_bedrooms,
            "num_bathrooms": num_bathrooms,
            "num_garage_spaces": num_garage_spaces,
            "is_furnished": is_furnished_binary
        }
        
    
        predicted_price = predict_price(data)
        st.write(f"Predicted Price: ${predicted_price:.2f}")
        
        
        if price > predicted_price:
            st.write("The property is priced higher than the predicted price.")
        elif abs(price - predicted_price) <= 70:  # we can change
            st.write("The property is priced close to the predicted price.")
        else:
            st.write("The property is priced lower than the predicted price.")

    
    if st.button("Restart"):
        os.system('killall streamlit')
        os.system('streamlit run your_app_name.py')

if __name__ == "__main__":
    main()

