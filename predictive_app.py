import time
import joblib
import pandas as pd
import streamlit as st

from log_utils import log_prediction

st.set_page_config(page_title="Units Sold Forecasting App (v1 vs v2)", layout="centered")
st.title("Units Sold Forecasting App (with Monitoring)")

@st.cache_resource
def load_models():
    model_v1 = joblib.load("units_sold_model_v1.pkl")
    model_v2 = joblib.load("units_sold_model_v2.pkl")
    return model_v1, model_v2

model_v1, model_v2 = load_models()

if "pred_ready" not in st.session_state:
    st.session_state["pred_ready"] = False
if "pred_v1" not in st.session_state:
    st.session_state["pred_v1"] = None
if "pred_v2" not in st.session_state:
    st.session_state["pred_v2"] = None
if "lat_v1" not in st.session_state:
    st.session_state["lat_v1"] = None
if "lat_v2" not in st.session_state:
    st.session_state["lat_v2"] = None

# Inputs
st.sidebar.header("Input Parameters")
price = st.sidebar.number_input("Price", min_value=0.0, value=100.0, step=1.0)
discount_pct = st.sidebar.slider("Discount (%)", 0.0, 100.0, 10.0, 1.0)

product_category = st.sidebar.selectbox(
    "Product Category",
    ["Sports", "Toys", "Home Decor", "Fashion", "Electronics"]
)
customer_segment = st.sidebar.selectbox(
    "Customer Segment",
    ["Occasional", "Premium", "Regular"]
)

net_price = price * (1 - discount_pct / 100.0)

input_df = pd.DataFrame({
    "net_price": [net_price],
    "Discount": [discount_pct],
    "Product_Category": [product_category],
    "Customer_Segment": [customer_segment],
})

st.subheader("Input Summary")
st.write(input_df)

if st.button("Run Prediction"):
    # v1 latency
    t0 = time.perf_counter()
    X_v1 = pd.DataFrame({"net_price": [net_price], "Discount": [discount_pct]})
    pred_v1 = model_v1.predict(X_v1)[0]
    lat_v1 = (time.perf_counter() - t0) * 1000.0

    # v2 latency
    t1 = time.perf_counter()
    X_v2 = input_df[["net_price", "Discount", "Product_Category", "Customer_Segment"]]
    pred_v2 = model_v2.predict(X_v2)[0]
    lat_v2 = (time.perf_counter() - t1) * 1000.0

    st.session_state["pred_v1"] = float(pred_v1)
    st.session_state["pred_v2"] = float(pred_v2)
    st.session_state["lat_v1"] = float(lat_v1)
    st.session_state["lat_v2"] = float(lat_v2)
    st.session_state["pred_ready"] = True

if st.session_state["pred_ready"]:
    st.subheader("Predictions (Units Sold)")
    st.write(f"**Model v1:** {st.session_state['pred_v1']:.2f} units  |  Latency: {st.session_state['lat_v1']:.2f} ms")
    st.write(f"**Model v2:** {st.session_state['pred_v2']:.2f} units  |  Latency: {st.session_state['lat_v2']:.2f} ms")
else:
    st.info("Click **Run Prediction** first.")

# Feedback + actual (optional)
st.subheader("Feedback & Actual Value (Optional)")
feedback_score = st.slider("Usefulness (1=Poor, 5=Excellent)", 1, 5, 4)
feedback_text = st.text_area("Comments (optional)")

actual_units = st.number_input(
    "Actual Units Sold (optional, if known)",
    min_value=0.0,
    value=0.0,
    step=1.0
)
actual_units_sold = None if actual_units == 0.0 else float(actual_units)

if st.button("Submit Feedback"):
    if not st.session_state["pred_ready"]:
        st.warning("Run prediction first.")
    else:
        # Log v1
        log_prediction(
            model_version="v1",
            price=price,
            discount_pct=discount_pct,
            product_category=product_category,
            customer_segment=customer_segment,
            units_sold_pred=st.session_state["pred_v1"],
            latency_ms=st.session_state["lat_v1"],
            feedback_score=feedback_score,
            feedback_text=feedback_text,
            actual_units_sold=actual_units_sold,
        )

        # Log v2
        log_prediction(
            model_version="v2",
            price=price,
            discount_pct=discount_pct,
            product_category=product_category,
            customer_segment=customer_segment,
            units_sold_pred=st.session_state["pred_v2"],
            latency_ms=st.session_state["lat_v2"],
            feedback_score=feedback_score,
            feedback_text=feedback_text,
            actual_units_sold=actual_units_sold,
        )

        st.success("Logged predictions + latency + feedback (and actual if provided).")


