import streamlit as st
import pandas as pd
import datetime
import pickle
import pydeck as pdk

# Predefined cities with lat/lon
city_coords = {
    'Bangalore': (12.9716, 77.5946),
    'Delhi': (28.6139, 77.2090),
    'Mumbai': (19.0760, 72.8777),
    'Chennai': (13.0827, 80.2707),
    'Kolkata': (22.5726, 88.3639)
}

@st.cache_resource
def load_model(path='model/weather_model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title("Weather Prediction by Location and Date")

# Select city
city = st.selectbox("Select City", list(city_coords.keys()))

# Select date
date = st.date_input("Select Date", datetime.date.today())

if st.button("Predict Temperature"):
    lat, lon = city_coords[city]
    dayofyear = date.timetuple().tm_yday

    # Prepare features for prediction
    features = [[lat, lon, dayofyear]]

    prediction = model.predict(features)[0]

    st.write(f"**Predicted Temperature for {city} on {date}:** {prediction:.2f} °C")

    # Show location on map
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/streets-v11',
        initial_view_state=pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=10,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=pd.DataFrame({'lat':[lat], 'lon':[lon]}),
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=10000,
            ),
        ],
    ))
