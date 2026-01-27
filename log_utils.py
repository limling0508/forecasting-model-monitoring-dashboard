import os
from datetime import datetime
import pandas as pd

# Always write logs next to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "monitoring_logs.csv")


def log_prediction(
    model_version: str,
    price: float,
    discount_pct: float,
    marketing_spend: float,
    product_category: str,
    customer_segment: str,
    units_sold_pred: float,
    latency_ms: float = None,
    feedback_score: int = None,
    feedback_text: str = None,
):
    """
    Append one prediction event to monitoring_logs.csv.
    Creates the file with header if it does not exist yet.
    """

    input_summary = (
        f"Price={price}, Discount%={discount_pct}, Marketing={marketing_spend}, "
        f"Category={product_category}, Segment={customer_segment}"
    )

    row = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "model_version": model_version,          # e.g., "v1" or "v2"
        "price": float(price),
        "discount_pct": float(discount_pct),
        "marketing_spend": float(marketing_spend),
        "product_category": str(product_category),
        "customer_segment": str(customer_segment),
        "input_summary": input_summary,
        "units_sold_pred": float(units_sold_pred),
        "latency_ms": float(latency_ms) if latency_ms is not None else None,
        "feedback_score": int(feedback_score) if feedback_score is not None else None,  # e.g., 1-5
        "feedback_text": feedback_text or "",
    }

    df_new = pd.DataFrame([row])

    if not os.path.exists(LOG_PATH):
        df_new.to_csv(LOG_PATH, index=False)
    else:
        df_new.to_csv(LOG_PATH, mode="a", header=False, index=False)
