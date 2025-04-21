import streamlit as st
import pandas as pd
import boto3
import psycopg2
from sqlalchemy import create_engine
import json

# --- Location Mapping ---
LOCATION_NAMES = {
    43: "Times Square",
    24: "Harlem",
    50: "Central Park"
}

# --- Cached Loaders ---
@st.cache_data(ttl=3600)  # Invalidate cache every hour
def load_athena_predictions():
    s3 = boto3.client("s3")
    bucket = "nyc-taxi-projects"
    prefix = "taxi/predictions/"
    all_preds = []

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("prediction.json"):
                response = s3.get_object(Bucket=bucket, Key=obj["Key"])
                record = json.loads(response["Body"].read())
                path_parts = obj["Key"].split("/")
                for part in path_parts:
                    if part.startswith("model="):
                        record["model"] = int(part.split("=")[1])
                    elif part.startswith("location_id="):
                        record["pickup_location_id"] = int(part.split("=")[1])
                    elif part.startswith("year="):
                        record["year"] = int(part.split("=")[1])
                    elif part.startswith("month="):
                        record["month"] = int(part.split("=")[1])
                    elif part.startswith("day="):
                        record["day"] = int(part.split("=")[1])
                    elif part.startswith("hour="):
                        record["hour"] = int(part.split("=")[1])
                all_preds.append(record)

    return pd.DataFrame(all_preds)


@st.cache_data(ttl=3600)
def load_rds_predictions():
    engine = create_engine(
        "postgresql://postgres:SaiRishi123@nyc-taxi-pred-db.csls0kaqy564.us-east-1.rds.amazonaws.com:5432/postgres"
    )
    df = pd.read_sql("SELECT * FROM predicted_rides", engine)
    return df

# --- Metrics ---
def compute_metrics(df):
    mae = (df["actual"] - df["predicted"]).abs().mean()
    mape = ((df["actual"] - df["predicted"]).abs() / df["actual"]).replace([float('inf')], 0).mean() * 100
    return round(mae, 2), round(mape, 2)

# --- Streamlit Layout ---
st.title("🚖 NYC Taxi Demand Forecast Dashboard")
tab1, tab2 = st.tabs(["🔍 Athena", "🗄️ RDS"])

with tab1:
    st.subheader("Predictions from Athena (S3)")
    athena_df = load_athena_predictions()

    if athena_df.empty:
        st.warning("No data found in Athena.")
    else:
        selected_loc = st.selectbox("Select Location", list(LOCATION_NAMES.keys()), format_func=lambda x: LOCATION_NAMES[x])
        filtered = athena_df[athena_df["pickup_location_id"] == selected_loc]

        for model_id in [1, 2]:
            model_df = filtered[filtered["model"] == model_id]
            if not model_df.empty:
                st.markdown(f"### Model {model_id}")
                mae, mape = compute_metrics(model_df.rename(columns={"predicted_trip_count": "predicted"}))
                st.metric("MAE", mae)
                st.metric("MAPE", f"{mape:.2f}%")

with tab2:
    st.subheader("Predictions from PostgreSQL (RDS)")
    rds_df = load_rds_predictions()

    if rds_df.empty:
        st.warning("No data found in RDS.")
    else:
        selected_loc_rds = st.selectbox("Select Location (RDS)", list(LOCATION_NAMES.keys()), format_func=lambda x: LOCATION_NAMES[x], key="rds_loc")
        filtered_rds = rds_df[rds_df["pickup_location_id"] == selected_loc_rds]

        for model_id in [1, 2]:
            model_df = filtered_rds[filtered_rds["model"] == model_id]
            if not model_df.empty:
                st.markdown(f"### Model {model_id}")
                mae, mape = compute_metrics(model_df.rename(columns={"predicted_trip_count": "predicted"}))
                st.metric("MAE", mae)
                st.metric("MAPE", f"{mape:.2f}%")




