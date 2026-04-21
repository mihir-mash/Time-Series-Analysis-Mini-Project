import os
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="IoT Anomaly Dashboard", layout="wide")
st.title("IoT Anomaly Detection Dashboard")


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def construct_datetime(df: pd.DataFrame) -> pd.Series:
    # Prefer original date column if present.
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce", utc=True)
        if dt.notna().sum() > 0:
            return dt

    # If day appears to be day-of-year (common in your notebook), build from year + day + hour.
    if all(c in df.columns for c in ["year", "day", "hour"]):
        day_numeric = pd.to_numeric(df["day"], errors="coerce")
        year_numeric = pd.to_numeric(df["year"], errors="coerce")
        hour_numeric = pd.to_numeric(df["hour"], errors="coerce").fillna(0)

        if day_numeric.notna().sum() > 0 and day_numeric.max() <= 366:
            base = pd.to_datetime(year_numeric.astype("Int64").astype(str) + "-01-01", errors="coerce", utc=True)
            return base + pd.to_timedelta(day_numeric - 1, unit="D") + pd.to_timedelta(hour_numeric, unit="h")

    # Fallback: try year + month + day + hour where month may be text (Jan, Feb, ...).
    if all(c in df.columns for c in ["year", "month", "day", "hour"]):
        m = df["month"].astype(str)
        month_map = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        month_num = m.str.strip().str[:3].str.lower().map(month_map)
        month_num = month_num.fillna(pd.to_numeric(m, errors="coerce"))

        temp = pd.DataFrame(
            {
                "year": pd.to_numeric(df["year"], errors="coerce"),
                "month": month_num,
                "day": pd.to_numeric(df["day"], errors="coerce"),
                "hour": pd.to_numeric(df["hour"], errors="coerce").fillna(0),
            }
        )
        dt = pd.to_datetime(
            temp[["year", "month", "day"]],
            errors="coerce",
            utc=True,
        ) + pd.to_timedelta(temp["hour"], unit="h")
        return dt

    # Last resort: synthetic timeline from row order.
    return pd.date_range("2020-01-01", periods=len(df), freq="h", tz="UTC")


def choose_sensor_columns(df: pd.DataFrame) -> List[str]:
    blocked = {"is_anomaly", "datetime", "index", "anomaly_roll", "mae"}
    candidates = [c for c in df.columns if c not in blocked]
    numeric_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols


required_files = ["submission.csv", "data_with_time.csv", "model_stats.csv"]
missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    st.error("Missing required files in current folder: " + ", ".join(missing))
    st.info("Run the Python notebook first so it generates submission.csv, data_with_time.csv, and model_stats.csv.")
    st.stop()

submission = load_csv("submission.csv")
df_raw = load_csv("data_with_time.csv")
model_stats = load_csv("model_stats.csv")

# Align output rows to anomaly flags. If notebook uses SEQ_LEN offset, infer it automatically.
if len(df_raw) > len(submission):
    seq_len_inferred = len(df_raw) - len(submission)
    df_plot = df_raw.iloc[seq_len_inferred:].copy().reset_index(drop=True)
else:
    seq_len_inferred = 0
    df_plot = df_raw.copy().reset_index(drop=True)

if len(df_plot) != len(submission):
    min_len = min(len(df_plot), len(submission))
    df_plot = df_plot.iloc[:min_len].copy()
    submission = submission.iloc[:min_len].copy()

df_plot["is_anomaly"] = submission["is_anomaly"].astype(int).values
df_plot["datetime"] = construct_datetime(df_plot)
df_plot = df_plot.sort_values("datetime").reset_index(drop=True)

sensor_cols = choose_sensor_columns(df_plot)
if not sensor_cols:
    st.error("No numeric sensor columns found for plotting.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    sensor_default_index = sensor_cols.index("Tpot") if "Tpot" in sensor_cols else 0
    sensor = st.selectbox("Select Sensor", options=sensor_cols, index=sensor_default_index)

    dt_min = df_plot["datetime"].min()
    dt_max = df_plot["datetime"].max()

    start_date = st.date_input("Start date", value=dt_min.date(), min_value=dt_min.date(), max_value=dt_max.date())
    end_date = st.date_input("End date", value=dt_max.date(), min_value=dt_min.date(), max_value=dt_max.date())

    only_anomalies = st.checkbox("Show only anomalies in table", value=True)

    st.markdown("---")
    st.subheader("Model Summary")

    threshold = float(model_stats["threshold"].iloc[0]) if "threshold" in model_stats.columns else np.nan
    total_points = int(model_stats["total_points"].iloc[0]) if "total_points" in model_stats.columns else len(df_plot)
    anomalies_count = int(model_stats["anomalies"].iloc[0]) if "anomalies" in model_stats.columns else int(df_plot["is_anomaly"].sum())
    anomaly_rate = float(model_stats["anomaly_rate"].iloc[0]) if "anomaly_rate" in model_stats.columns else float(df_plot["is_anomaly"].mean())

    st.metric("Threshold (MAE)", f"{threshold:.4f}" if np.isfinite(threshold) else "N/A")
    st.metric("Total Points", f"{total_points}")
    st.metric("Anomalies", f"{anomalies_count}")
    st.metric("Anomaly Rate", f"{anomaly_rate * 100:.2f}%")
    st.caption(f"Inferred SEQ_LEN offset: {seq_len_inferred}")


start_dt = pd.Timestamp(start_date, tz="UTC")
end_dt = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

filtered = df_plot[(df_plot["datetime"] >= start_dt) & (df_plot["datetime"] <= end_dt)].copy()

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader(f"Sensor Trend: {sensor}")

    fig = px.line(filtered, x="datetime", y=sensor, title=f"{sensor} (Red points = anomalies)")
    anomalies = filtered[filtered["is_anomaly"] == 1]
    if len(anomalies) > 0:
        fig.add_scatter(
            x=anomalies["datetime"],
            y=anomalies[sensor],
            mode="markers",
            marker={"color": "red", "size": 7},
            name="Anomaly",
        )
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Current Range")
    st.write("Rows:", len(filtered))
    st.write("Anomalies:", int(filtered["is_anomaly"].sum()))
    st.write("Anomaly %:", f"{100 * filtered['is_anomaly'].mean():.2f}%" if len(filtered) else "N/A")

st.subheader("Detected Records")
show_df = filtered.copy()
if only_anomalies:
    show_df = show_df[show_df["is_anomaly"] == 1]

display_cols = ["datetime", sensor, "is_anomaly"]
if "mae" in show_df.columns:
    display_cols.append("mae")

st.dataframe(show_df[display_cols].sort_values("datetime"), use_container_width=True)

csv_bytes = show_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered data as CSV",
    data=csv_bytes,
    file_name="filtered_anomalies.csv",
    mime="text/csv",
)
