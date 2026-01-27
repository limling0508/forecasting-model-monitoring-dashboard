import os
from datetime import datetime
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "monitoring_logs.csv")


def log_prediction(
    model_version: str,
    price: float,
    discount_pct: float,
    product_category: str,
    customer_segment: str,
    units_sold_pred: float,
    latency_ms: float = None,
    feedback_score: int = None,
    feedback_text: str = None,
    actual_units_sold: float = None,   # NEW
):
    """
    Append one prediction event to monitoring_logs.csv.
    If actual_units_sold is provided, error columns are computed.
    """

    input_summary = (
        f"Price={price}, Discount%={discount_pct}, "
        f"Category={product_category}, Segment={customer_segment}"
    )

    abs_error = None
    squared_error = None
    if actual_units_sold is not None:
        abs_error = abs(float(actual_units_sold) - float(units_sold_pred))
        squared_error = (float(actual_units_sold) - float(units_sold_pred)) ** 2

    row = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "model_version": model_version,  # "v1" or "v2"

        "price": float(price),
        "discount_pct": float(discount_pct),
        "product_category": str(product_category),
        "customer_segment": str(customer_segment),

        "input_summary": input_summary,

        "units_sold_pred": float(units_sold_pred),
        "actual_units_sold": float(actual_units_sold) if actual_units_sold is not None else None,  # NEW
        "abs_error": float(abs_error) if abs_error is not None else None,  # NEW
        "squared_error": float(squared_error) if squared_error is not None else None,  # NEW

        "latency_ms": float(latency_ms) if latency_ms is not None else None,

        "feedback_score": int(feedback_score) if feedback_score is not None else None,
        "feedback_text": feedback_text or "",
    }

    df_new = pd.DataFrame([row])

    if not os.path.exists(LOG_PATH):
        df_new.to_csv(LOG_PATH, index=False)
    else:
        df_new.to_csv(LOG_PATH, mode="a", header=False, index=False)


