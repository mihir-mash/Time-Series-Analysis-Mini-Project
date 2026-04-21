# Weather IoT Anomaly Detection (Mini Project)

A compact time-series anomaly detection project using an LSTM autoencoder on weather IoT data.

## What this project contains

- `weather-iot-anomaly-detection-lstm-python.ipynb`: Full workflow (EDA, preprocessing, model training, anomaly scoring, reports).
- `app.py`: Streamlit dashboard to explore detected anomalies.
- `archive/cleaned_weather.csv`: Input dataset.
- `submission.csv`: Final anomaly labels (`is_anomaly`) used by the dashboard.
- `data_with_time.csv`: Time-aligned data used for plotting/filtering.
- `model_stats.csv`: Summary stats (threshold, anomaly rate, etc.).
- `top_anomalies.csv`: Highest anomaly-score records.
- `loss_plot.png`, `Tpot_plot.png`, `evaluation_report.png`: Generated visuals.

## Quick start

1. Create/activate a Python environment.
2. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn tensorflow streamlit
```

3. Run the notebook `weather-iot-anomaly-detection-lstm-python.ipynb` from top to bottom.
4. This generates the required CSV outputs (`submission.csv`, `data_with_time.csv`, `model_stats.csv`).
5. Launch the dashboard:

```bash
streamlit run app.py
```

## Dashboard features

- Select sensor column to visualize.
- Date-range filtering.
- Anomaly markers over time-series plots.
- Summary metrics from `model_stats.csv`.
- Filtered anomaly table + CSV download.

## Notes

- Keep `app.py` and generated CSV files in the same folder.
- If the app says files are missing, rerun the notebook cells that save outputs.
- TensorFlow can be slow on CPU-only systems; GPU support is optional.