# log_utils.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "monitoring_logs.csv"


def log_prediction(
    model_version,
    price,
    discount_pct,
    product_category,
    customer_segment,
    units_sold_pred,
    latency_ms=None,
    feedback_score=None,
    feedback_text=None,
    actual_units_sold=None,
):
    row = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "model_version": str(model_version),
        "price": float(price),
        "discount_pct": float(discount_pct),
        "product_category": str(product_category),
        "customer_segment": str(customer_segment),
        "units_sold_pred": float(units_sold_pred),
        "latency_ms": float(latency_ms) if latency_ms is not None else None,
        "feedback_score": int(feedback_score) if feedback_score is not None else None,
        "feedback_text": (feedback_text or "").strip(),
        "actual_units_sold": float(actual_units_sold) if actual_units_sold is not None else None,
    }

    df_new = pd.DataFrame([row])

    if not LOG_PATH.exists():
        df_new.to_csv(LOG_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    else:
        df_new.to_csv(LOG_PATH, mode="a", header=False, index=False, quoting=csv.QUOTE_MINIMAL)

