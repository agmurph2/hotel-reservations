# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import urllib.request
import os
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Hotel Booking Cancellation Prediction", layout="wide")

st.title("Hotel Booking Cancellation — Streamlit Demo")
st.markdown( 
     """
This app reconstructs the exact feature vector (one-hot columns included) used by your trained model
and returns a cancellation prediction + probability.
"""
)


MODEL_URL = "https://raw.githubusercontent.com/augie480/CIS412teamproject/main/hotel_model.pkl"
MODEL_PATH = "hotel_model.pkl"

SCALER_URL = "https://raw.githubusercontent.com/augie480/CIS412teamproject/main/scaler.pkl"
SCALER_PATH = "scaler.pkl"


if not os.path.exists(MODEL_PATH):
    try:
        with st.spinner("Downloading model from GitHub..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model downloaded.")
    except Exception as e:
        st.error(f"Could not download model from {MODEL_URL}. Error: {e}")


scaler_loaded = False
scaler = None
if not os.path.exists(SCALER_PATH):
    try:
        urllib.request.urlretrieve(SCALER_URL, SCALER_PATH)
        st.info("Scaler downloaded from GitHub.")
    except Exception:
        st.warning(
            "No scaler found at the default URL. If you used StandardScaler at training time, "
            "please save it (`joblib.dump(scaler, 'scaler.pkl')`) to your GitHub repo and update SCALER_URL."
        )

if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
        scaler_loaded = True
        st.success("Scaler loaded.")
    except Exception as e:
        st.warning(f"Scaler file exists but could not be loaded: {e}")

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        st.success("Model loaded.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
else:
    st.error("Model file not available locally. Please ensure MODEL_URL points to a raw .pkl on GitHub.")


FEATURE_COLUMNS = [
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "required_car_parking_space",
    "lead_time",
    "arrival_year",
    "arrival_month",
    "arrival_date",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests",
    "type_of_meal_plan_Meal Plan 2",
    "type_of_meal_plan_Meal Plan 3",
    "type_of_meal_plan_Not Selected",
    "room_type_reserved_Room_Type 2",
    "room_type_reserved_Room_Type 3",
    "room_type_reserved_Room_Type 4",
    "room_type_reserved_Room_Type 5",
    "room_type_reserved_Room_Type 6",
    "room_type_reserved_Room_Type 7",
    "market_segment_type_Complementary",
    "market_segment_type_Corporate",
    "market_segment_type_Offline",
    "market_segment_type_Online",
]


NUMERIC_COLS = [
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "required_car_parking_space",
    "lead_time",
    "arrival_year",
    "arrival_month",
    "arrival_date",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests",
]

st.header("Reservation Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    no_of_adults = st.number_input("No. of adults", min_value=0, value=1, step=1)
    no_of_children = st.number_input("No. of children", min_value=0, value=0, step=1)
    no_of_weekend_nights = st.number_input("Weekend nights", min_value=0, value=0, step=1)
    no_of_week_nights = st.number_input("Week nights", min_value=0, value=1, step=1)
    required_car_parking_space = st.number_input("Required car parking space", min_value=0, value=0, step=1)

with col2:
    lead_time = st.number_input("Lead time (days)", min_value=0, value=30, step=1)
    arrival_year = st.number_input("Arrival year", min_value=2000, max_value=2100, value=2024, step=1)
    arrival_month = st.selectbox(
        "Arrival month",
        options=list(range(1, 13)),
        index=0
    )
    arrival_date = st.number_input("Arrival date (day of month)", min_value=1, max_value=31, value=1, step=1)
    repeated_guest = st.selectbox("Repeated guest?", options=[0, 1], index=0)

with col3:
    no_of_previous_cancellations = st.number_input("Previous cancellations", min_value=0, value=0, step=1)
    no_of_previous_bookings_not_canceled = st.number_input("Previous bookings not canceled", min_value=0, value=0, step=1)
    avg_price_per_room = st.number_input("Average price per room", min_value=0.0, value=100.0, step=1.0, format="%.2f")
    no_of_special_requests = st.number_input("No. of special requests", min_value=0, value=0, step=1)

st.markdown("---")
st.subheader("Categorical choices (will be converted to one-hot)")

col4, col5, col6 = st.columns(3)

with col4:
    meal_plan = st.selectbox(
        "Meal plan",
        options=[
            "Meal Plan 1 (base)",
            "Meal Plan 2",
            "Meal Plan 3",
            "Not Selected"
        ],
        index=0
    )

with col5:
    room_type = st.selectbox(
        "Room type reserved",
        options=[
            "Room_Type 1 (base)",
            "Room_Type 2",
            "Room_Type 3",
            "Room_Type 4",
            "Room_Type 5",
            "Room_Type 6",
            "Room_Type 7"
        ],
        index=0
    )

with col6:
    market_segment = st.selectbox(
        "Market segment",
        options=[
            "Other (base)",
            "Complementary",
            "Corporate",
            "Offline",
            "Online"
        ],
        index=0
    )

st.markdown("---")
st.write(
    " Note: The code assumes your model used the feature names you provided (one-hot columns listed). "
    "If you exported a `scaler.pkl` at training time, put it in your GitHub repo and update `SCALER_URL`."
)


input_vector = {c: 0 for c in FEATURE_COLUMNS}


input_vector["no_of_adults"] = no_of_adults
input_vector["no_of_children"] = no_of_children
input_vector["no_of_weekend_nights"] = no_of_weekend_nights
input_vector["no_of_week_nights"] = no_of_week_nights
input_vector["required_car_parking_space"] = required_car_parking_space
input_vector["lead_time"] = lead_time
input_vector["arrival_year"] = arrival_year
input_vector["arrival_month"] = arrival_month
input_vector["arrival_date"] = arrival_date
input_vector["repeated_guest"] = repeated_guest
input_vector["no_of_previous_cancellations"] = no_of_previous_cancellations
input_vector["no_of_previous_bookings_not_canceled"] = no_of_previous_bookings_not_canceled
input_vector["avg_price_per_room"] = avg_price_per_room
input_vector["no_of_special_requests"] = no_of_special_requests


if meal_plan == "Meal Plan 2":
    input_vector["type_of_meal_plan_Meal Plan 2"] = 1
elif meal_plan == "Meal Plan 3":
    input_vector["type_of_meal_plan_Meal Plan 3"] = 1
elif meal_plan == "Not Selected":
    input_vector["type_of_meal_plan_Not Selected"] = 1

if room_type == "Room_Type 2":
    input_vector["room_type_reserved_Room_Type 2"] = 1
elif room_type == "Room_Type 3":
    input_vector["room_type_reserved_Room_Type 3"] = 1
elif room_type == "Room_Type 4":
    input_vector["room_type_reserved_Room_Type 4"] = 1
elif room_type == "Room_Type 5":
    input_vector["room_type_reserved_Room_Type 5"] = 1
elif room_type == "Room_Type 6":
    input_vector["room_type_reserved_Room_Type 6"] = 1
elif room_type == "Room_Type 7":
    input_vector["room_type_reserved_Room_Type 7"] = 1

if market_segment == "Complementary":
    input_vector["market_segment_type_Complementary"] = 1
elif market_segment == "Corporate":
    input_vector["market_segment_type_Corporate"] = 1
elif market_segment == "Offline":
    input_vector["market_segment_type_Offline"] = 1
elif market_segment == "Online":
    input_vector["market_segment_type_Online"] = 1

input_df = pd.DataFrame([input_vector], columns=FEATURE_COLUMNS)


if scaler_loaded and scaler is not None:
    try:
        
        numeric_idx = [input_df.columns.get_loc(c) for c in NUMERIC_COLS if c in input_df.columns]
        
        arr = input_df.values.astype(float)
        arr[:, numeric_idx] = scaler.transform(arr[:, numeric_idx])
        input_df = pd.DataFrame(arr, columns=input_df.columns)
        st.info("Numeric inputs scaled using loaded scaler.")
    except Exception as e:
        st.warning(f"Scaler exists but scaling failed: {e}. Proceeding without scaling.")

else:
    st.warning("No scaler loaded — numeric features will NOT be scaled. This may affect predictions if the model expects scaled inputs.")


with st.expander("Show full model input vector"):
    st.write(input_df.T)


if st.button("Predict cancellation"):
    if model is None:
        st.error("Model not loaded - cannot predict.")
    else:
        try:
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

            st.success("Prediction done.")
            st.write("### Result")
            st.write("**Canceled**" if pred == 1 else "**Not Canceled**")
            if proba is not None:
                st.write(f"**Probability of cancellation:** {proba:.3f}")

            
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
                fi_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": fi}).sort_values("importance", ascending=False).head(10)
                st.write("Top feature importances (model-provided)")
                st.table(fi_df.reset_index(drop=True))
        except Exception as e:
            st.error(f"Prediction failed: {e}")


st.markdown("---")
st.write(
    "Deployment notes:\n\n"
    "- If your model expected scaled numeric inputs, save the scaler at training time with `joblib.dump(scaler, 'scaler.pkl')` and place it in the same GitHub repo; update `SCALER_URL` above.\n"
    "- If your model name or GitHub location differ, update `MODEL_URL`.\n"
    "- To run locally: `pip install -r requirements.txt` (or at least streamlit, scikit-learn, joblib, pandas) and `streamlit run app.py`.\n"
)
