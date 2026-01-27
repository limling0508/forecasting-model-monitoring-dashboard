# monitor_dashboard.py
import os
import pandas as pd
import streamlit as st

from log_utils import LOG_PATH

st.set_page_config(page_title="Model Monitoring & Feedback", layout="wide")
st.title("Model Monitoring & Feedback Dashboard")


@st.cache_data
def load_logs():
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()

    df = pd.read_csv(LOG_PATH)

    # Parse timestamp
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")

    # Sort by time (safe even if some NaT)
    if "timestamp_utc" in df.columns:
        df = df.sort_values("timestamp_utc")

    return df


logs = load_logs()

# Handle "no logs yet"
if logs.empty:
    st.warning(
        "No monitoring logs found yet. "
        "Run the prediction app and submit feedback at least once, then refresh this page."
    )
    st.stop()

# ---------- Sidebar Filters ----------
st.sidebar.header("Filters")

# Normalize casing (helps matching dropdown options)
if "product_category" in logs.columns:
    logs["product_category"] = logs["product_category"].astype(str).str.strip().str.title()
if "customer_segment" in logs.columns:
    logs["customer_segment"] = logs["customer_segment"].astype(str).str.strip().str.title()

models = ["All"] + sorted(logs["model_version"].dropna().unique().tolist())
selected_model = st.sidebar.selectbox("Model version", models)

# Fixed dropdown options
category_options = ["All", "Sports", "Toys", "Home Decor", "Fashion", "Electronics"]
segment_options = ["All", "Occasional", "Premium", "Regular"]

selected_category = st.sidebar.selectbox("Product category", category_options)
selected_segment = st.sidebar.selectbox("Customer segment", segment_options)

filtered = logs.copy()
if selected_model != "All":
    filtered = filtered[filtered["model_version"] == selected_model]
if selected_category != "All":
    filtered = filtered[filtered["product_category"] == selected_category]
if selected_segment != "All":
    filtered = filtered[filtered["customer_segment"] == selected_segment]

# ---------- Key Metrics ----------
st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Predictions", len(filtered))

if "feedback_score" in filtered.columns and filtered["feedback_score"].notna().any():
    col2.metric("Avg Feedback Score", f"{filtered['feedback_score'].mean():.2f}")
else:
    col2.metric("Avg Feedback Score", "N/A")

if "latency_ms" in filtered.columns and filtered["latency_ms"].notna().any():
    col3.metric("Avg Latency (ms)", f"{filtered['latency_ms'].mean():.2f}")
else:
    col3.metric("Avg Latency (ms)", "N/A")

if "units_sold_pred" in filtered.columns and filtered["units_sold_pred"].notna().any():
    col4.metric("Avg Predicted Units", f"{filtered['units_sold_pred'].mean():.2f}")
else:
    col4.metric("Avg Predicted Units", "N/A")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Monitoring Overview", "Feedback Analysis", "Raw Logs"])

# ---------- TAB 1: Monitoring Overview ----------
with tab1:
    st.subheader("Prediction Trend Over Time")
    if "timestamp_utc" in filtered.columns and filtered["timestamp_utc"].notna().any():
        trend = (
            filtered.dropna(subset=["timestamp_utc"])
            .set_index("timestamp_utc")[["units_sold_pred"]]
        )
        st.line_chart(trend)
    else:
        st.info("No valid timestamps to plot trend.")

    st.subheader("Predicted Units Sold Distribution (by Model Version)")
    if "model_version" in filtered.columns and "units_sold_pred" in filtered.columns:
        dist = filtered[["model_version", "units_sold_pred"]].dropna()
        if dist.empty:
            st.info("No prediction values to summarise.")
        else:
            summary = (
                dist.groupby("model_version")["units_sold_pred"]
                .agg(["count", "mean", "min", "max"])
                .reset_index()
            )
            st.dataframe(summary)
    else:
        st.info("Missing columns for prediction distribution.")

    st.subheader("Input Monitoring (Price & Discount%)")
    c1, c2 = st.columns(2)
    with c1:
        if "price" in filtered.columns and filtered["price"].notna().any():
            st.write("Price distribution (top 30 values)")
            st.bar_chart(filtered["price"].value_counts().sort_index().head(30))
        else:
            st.info("No price data available.")
    with c2:
        if "discount_pct" in filtered.columns and filtered["discount_pct"].notna().any():
            st.write("Discount% distribution (top 30 values)")
            st.bar_chart(filtered["discount_pct"].value_counts().sort_index().head(30))
        else:
            st.info("No discount% data available.")

    st.markdown("---")

    # ---------- Model Comparison: Latency + MAE ----------
    st.subheader("Model Comparison: Avg Latency and MAE")

    metrics_rows = []

    for mv in sorted(filtered["model_version"].dropna().unique()):
        sub_all = filtered[filtered["model_version"] == mv].copy()

        avg_latency = None
        if "latency_ms" in sub_all.columns and sub_all["latency_ms"].notna().any():
            avg_latency = sub_all["latency_ms"].mean()

        # MAE only when actual is available
        mae = None
        if "actual_units_sold" in sub_all.columns and "units_sold_pred" in sub_all.columns:
            sub_eval = sub_all.dropna(subset=["actual_units_sold", "units_sold_pred"]).copy()

            if not sub_eval.empty:
                # Prefer precomputed abs_error if present
                if "abs_error" in sub_eval.columns and sub_eval["abs_error"].notna().any():
                    mae = sub_eval["abs_error"].mean()
                else:
                    mae = (sub_eval["actual_units_sold"] - sub_eval["units_sold_pred"]).abs().mean()

        metrics_rows.append({
            "model_version": mv,
            "avg_latency_ms": avg_latency,
            "MAE": mae
        })

    metrics_table = pd.DataFrame(metrics_rows).set_index("model_version")

    st.dataframe(
        metrics_table.style.format({
            "avg_latency_ms": lambda x: "N/A" if pd.isna(x) else f"{x:.2f}",
            "MAE": lambda x: "N/A" if pd.isna(x) else f"{x:.2f}",
        })
    )

    st.caption("Note: MAE is shown only when Actual Units Sold values have been logged.")

# ---------- TAB 2: Feedback Analysis ----------
with tab2:
    st.subheader("Average Feedback Score by Model Version")
    if "feedback_score" in logs.columns:
        fb = logs.groupby("model_version")["feedback_score"].mean().reset_index()
        fb = fb.dropna(subset=["feedback_score"])

        if fb.empty:
            st.info("No feedback scores yet.")
        else:
            st.bar_chart(fb.set_index("model_version"))
    else:
        st.info("feedback_score column not found in logs.")

    st.subheader("Recent Comments")
    if "feedback_text" in logs.columns:
        comments = logs.copy()
        comments["feedback_text"] = comments["feedback_text"].astype(str)
        comments = comments[comments["feedback_text"].str.strip() != ""]
        if "timestamp_utc" in comments.columns:
            comments = comments.sort_values("timestamp_utc", ascending=False)
        comments = comments.head(10)

        if comments.empty:
            st.info("No qualitative comments yet.")
        else:
            for _, row in comments.iterrows():
                ts = row.get("timestamp_utc", "")
                st.write(f"**[{ts}] {row.get('model_version', '')} – Score: {row.get('feedback_score', 'N/A')}**")
                st.write(row["feedback_text"])
                st.markdown("---")
    else:
        st.info("feedback_text column not found in logs.")

# ---------- TAB 3: Raw Logs ----------
with tab3:
    st.subheader("Raw Monitoring Logs")
    st.dataframe(filtered)

