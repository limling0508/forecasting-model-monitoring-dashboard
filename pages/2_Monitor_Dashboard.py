# pages/2_Monitor_Dashboard.py
import os
import pandas as pd
import streamlit as st

from log_utils import LOG_PATH

st.set_page_config(page_title="Model Monitoring & Feedback", layout="wide")
st.title("Model Monitoring & Feedback Dashboard")

# ---- Manual refresh (solves caching not updating) ----
st.sidebar.header("Controls")
if st.sidebar.button("Refresh logs"):
    st.cache_data.clear()
    st.rerun()

# ---- Load logs (auto-refresh every 2 seconds) ----
@st.cache_data(ttl=2)
def load_logs(log_path: str):
    if not os.path.exists(log_path):
        return pd.DataFrame()

    df = pd.read_csv(log_path)

    # Parse timestamp
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")

    # Sort by time (safe even if some NaT)
    if "timestamp_utc" in df.columns:
        df = df.sort_values("timestamp_utc")

    return df


logs = load_logs(str(LOG_PATH))

# Handle "no logs yet"
if logs.empty:
    st.warning(
        "No monitoring logs found yet. "
        "Run the prediction app and submit feedback at least once, then click **Refresh logs**."
    )
    st.info(f"Looking for logs at: {LOG_PATH}")
    st.stop()

# ---------- Sidebar Filters ----------
st.sidebar.header("Filters")

# Normalize casing (helps matching dropdown options)
if "product_category" in logs.columns:
    logs["product_category"] = logs["product_category"].astype(str).str.strip().str.title()
if "customer_segment" in logs.columns:
    logs["customer_segment"] = logs["customer_segment"].astype(str).str.strip().str.title()

models = ["All"]
if "model_version" in logs.columns:
    models += sorted(logs["model_version"].dropna().unique().tolist())
selected_model = st.sidebar.selectbox("Model version", models)

# Fixed dropdown options
category_options = ["All", "Sports", "Toys", "Home Decor", "Fashion", "Electronics"]
segment_options = ["All", "Occasional", "Premium", "Regular"]

selected_category = st.sidebar.selectbox("Product category", category_options)
selected_segment = st.sidebar.selectbox("Customer segment", segment_options)

filtered = logs.copy()
if selected_model != "All" and "model_version" in filtered.columns:
    filtered = filtered[filtered["model_version"] == selected_model]
if selected_category != "All" and "product_category" in filtered.columns:
    filtered = filtered[filtered["product_category"] == selected_category]
if selected_segment != "All" and "customer_segment" in filtered.columns:
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
    if (
        "timestamp_utc" in filtered.columns
        and "units_sold_pred" in filtered.columns
        and filtered["timestamp_utc"].notna().any()
    ):
        trend = (
            filtered.dropna(subset=["timestamp_utc"])
            .set_index("timestamp_utc")[["units_sold_pred"]]
        )
        st.line_chart(trend)
    else:
        st.info("No valid timestamp/prediction data to plot trend.")

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

    st.subheader("Input Monitoring (Price & Discount%) — Histogram")
    c1, c2 = st.columns(2)

    with c1:
        if "price" in filtered.columns and filtered["price"].notna().any():
            st.write("Price histogram (bins=20)")
            price_series = filtered["price"].dropna().astype(float)
            price_bins = pd.cut(price_series, bins=20)
            price_hist = price_bins.value_counts().sort_index()

            price_df = price_hist.reset_index()
            price_df.columns = ["bin", "count"]
            price_df["bin"] = price_df["bin"].astype(str)  # IMPORTANT: Interval -> string

            st.bar_chart(price_df.set_index("bin"))
        else:
            st.info("No price data available.")

    with c2:
        if "discount_pct" in filtered.columns and filtered["discount_pct"].notna().any():
            st.write("Discount% histogram (bins=20)")
            disc_series = filtered["discount_pct"].dropna().astype(float)
            disc_bins = pd.cut(disc_series, bins=20)
            disc_hist = disc_bins.value_counts().sort_index()

            disc_df = disc_hist.reset_index()
            disc_df.columns = ["bin", "count"]
            disc_df["bin"] = disc_df["bin"].astype(str)  # IMPORTANT: Interval -> string

            st.bar_chart(disc_df.set_index("bin"))
        else:
            st.info("No discount% data available.")

    st.markdown("---")

    # ---------- Model Comparison: Latency + MAE ----------
    st.subheader("Model Comparison: Avg Latency and MAE")

    if "model_version" not in filtered.columns:
        st.info("model_version column not found, cannot compare models.")
    else:
        metrics_rows = []
        for mv in sorted(filtered["model_version"].dropna().unique()):
            sub_all = filtered[filtered["model_version"] == mv].copy()

            avg_latency = None
            if "latency_ms" in sub_all.columns and sub_all["latency_ms"].notna().any():
                avg_latency = sub_all["latency_ms"].mean()

            mae = None
            if "actual_units_sold" in sub_all.columns and "units_sold_pred" in sub_all.columns:
                sub_eval = sub_all.dropna(subset=["actual_units_sold", "units_sold_pred"]).copy()

                if not sub_eval.empty:
                    if "abs_error" in sub_eval.columns and sub_eval["abs_error"].notna().any():
                        mae = sub_eval["abs_error"].mean()
                    else:
                        mae = (sub_eval["actual_units_sold"] - sub_eval["units_sold_pred"]).abs().mean()

            metrics_rows.append({"model_version": mv, "avg_latency_ms": avg_latency, "MAE": mae})

        metrics_table = pd.DataFrame(metrics_rows).set_index("model_version")
        st.dataframe(
            metrics_table.style.format(
                {
                    "avg_latency_ms": lambda x: "N/A" if pd.isna(x) else f"{x:.2f}",
                    "MAE": lambda x: "N/A" if pd.isna(x) else f"{x:.2f}",
                }
            )
        )
        st.caption("Note: MAE is shown only when Actual Units Sold values have been logged.")

# ---------- TAB 2: Feedback Analysis ----------
with tab2:
    st.subheader("Average Feedback Score by Model Version")
    if "feedback_score" in logs.columns and "model_version" in logs.columns:
        fb = logs.groupby("model_version")["feedback_score"].mean().reset_index()
        fb = fb.dropna(subset=["feedback_score"])

        if fb.empty:
            st.info("No feedback scores yet.")
        else:
            st.bar_chart(fb.set_index("model_version"))
    else:
        st.info("feedback_score/model_version column not found in logs.")

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
                st.write(
                    f"**[{ts}] {row.get('model_version', '')} – Score: {row.get('feedback_score', 'N/A')}**"
                )
                st.write(row["feedback_text"])
                st.markdown("---")
    else:
        st.info("feedback_text column not found in logs.")

# ---------- TAB 3: Raw Logs ----------
with tab3:
    st.subheader("Raw Monitoring Logs")
    st.dataframe(filtered)
