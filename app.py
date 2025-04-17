
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="📈 Time Series Forecast App", layout="wide")
st.title("📈 Time Series Forecasting Web App")
st.markdown("Upload an Excel file with time series data. Select a sheet, choose decomposition & model, and view forecasts.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an Excel file", type=["xlsx"])

# Forecasting model and decomposition type selection
model_choice = st.selectbox("Choose Forecasting Model", ["ARIMA", "ETS", "Prophet"])
decomp_choice = st.radio("Choose Decomposition Type", ["Additive", "Multiplicative"])

# Evaluation metric function
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mse = mean_squared_error(y_true, y_pred)
    return rmse, mae, mape, mse

if uploaded_file:
    try:
        xlsx = pd.ExcelFile(uploaded_file)
        sheet = st.selectbox("📑 Select Sheet", xlsx.sheet_names)
        df = xlsx.parse(sheet)
        df.columns = df.columns.str.strip()

        date_col = "Date" if "Date" in df.columns else df.columns[0]
        value_col = df.columns[1]

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, value_col])
        df.set_index(date_col, inplace=True)
        ts = df[value_col].asfreq('B').fillna(method='ffill')

        st.subheader("📊 Raw Time Series")
        st.line_chart(ts)

        # Decomposition
        try:
            result = seasonal_decompose(ts, model=decomp_choice.lower(), period=30)
            st.subheader("🔍 Time Series Decomposition")
            fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
            result.observed.plot(ax=axs[0], title='Observed')
            result.trend.plot(ax=axs[1], title='Trend')
            result.seasonal.plot(ax=axs[2], title='Seasonality')
            result.resid.plot(ax=axs[3], title='Residuals')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Decomposition failed: {e}")

        # Forecasting
        train_size = int(len(ts) * 0.8)
        train, test = ts[:train_size], ts[train_size:]
        future_dates = pd.date_range(start=test.index[-1] + pd.Timedelta(days=1), periods=60, freq='B')

        st.subheader("🔮 Forecast Plot")

        if model_choice == "ARIMA":
            model = ARIMA(train, order=(5, 1, 2)).fit()
            forecast = model.forecast(steps=len(test))
            future_forecast = model.forecast(steps=60)

        elif model_choice == "ETS":
            model = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
            forecast = model.forecast(steps=len(test))
            future_forecast = model.forecast(steps=60)

        elif model_choice == "Prophet":
            prophet_df = ts.reset_index().rename(columns={ts.index.name: "ds", ts.name: "y"})
            prophet_model = Prophet(daily_seasonality=True)
            prophet_model.fit(prophet_df.iloc[:train_size])
            future = prophet_model.make_future_dataframe(periods=len(test) + 60, freq='B')
            forecast_df = prophet_model.predict(future)
            forecast = forecast_df['yhat'].iloc[train_size:train_size+len(test)].values
            future_forecast = forecast_df['yhat'].iloc[-60:].values

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ts.index, ts, label="Actual")
        ax.plot(test.index, forecast, label="Forecast")
        ax.plot(future_dates, future_forecast, label="Future Forecast (60 days)", linestyle="--")
        ax.set_title(f"{model_choice} Forecast")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Evaluation
        rmse, mae, mape, mse = evaluate(test, forecast)
        st.subheader("📏 Forecast Accuracy Metrics")
        st.write(f"**RMSE:** {rmse:.4f}")
        st.write(f"**MAE:** {mae:.4f}")
        st.write(f"**MAPE:** {mape:.2f}%")
        st.write(f"**MSE:** {mse:.4f}")

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")

# Footer in bottom-right corner
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 10px;
            right: 10px;
            font-size: 13px;
            color: gray;
        }
    </style>
    <div class="footer">🛠️ Designed by <strong>Parshva</strong></div>
    """,
    unsafe_allow_html=True
)
