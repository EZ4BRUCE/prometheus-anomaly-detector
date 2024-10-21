import logging
import pandas as pd
import numpy as np
from prometheus_api_client import Metric
from statsmodels.tsa.statespace.sarimax import SARIMAX
from engine.model_predictor.model_predictor import ModelPredictor
# Set up logging
_LOGGER = logging.getLogger(__name__)

class SarimaPredictor(ModelPredictor):
    """SARIMA model for time series forecasting."""

    model_name = "sarima"
    model_description = "Forecast value based on SARIMA model"
    model = None
    predicted_df = None
    metric = None
    label_config_with_matric_name = None
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)

    def __init__(self, metric, label_config_with_matric_name, rolling_data_window_size="10d", order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """Initialize the SARIMA model."""
        self.metric = Metric(metric, rolling_data_window_size)
        self.label_config_with_matric_name = label_config_with_matric_name
        self.order = order
        self.seasonal_order = seasonal_order

    def train(self, metric_data=None, prediction_duration=15):
        """Train the SARIMA model and store the predictions in pandas dataframe."""
        if metric_data:
            self.metric += Metric(metric_data)

        data = self.metric.metric_values
        vals = np.array(data["y"].tolist())

        _LOGGER.debug("Training data start time: %s", self.metric.start_time)
        _LOGGER.debug("Training data end time: %s", self.metric.end_time)
        _LOGGER.debug("Begin training")

        # Fit the SARIMA model
        model = SARIMAX(vals, order=self.order, seasonal_order=self.seasonal_order)
        model_fit = model.fit(disp=False)

        # Forecast future values
        forecast_values = model_fit.forecast(steps=prediction_duration)
        dataframe_cols = {"yhat": np.array(forecast_values)}

        # Calculate upper and lower bounds
        conf_int = model_fit.get_forecast(steps=prediction_duration).conf_int()
        dataframe_cols["yhat_upper"] = conf_int[:, 1]
        dataframe_cols["yhat_lower"] = conf_int[:, 0]

        # Create timestamps for the forecast
        maximum_time = max(data["ds"])
        dataframe_cols["timestamp"] = pd.date_range(
            maximum_time, periods=len(forecast_values), freq="min"
        )

        # Create a DataFrame for the forecast
        forecast = pd.DataFrame(data=dataframe_cols)
        forecast = forecast.set_index("timestamp")

        self.predicted_df = forecast
        _LOGGER.debug(forecast)

    def predict_value(self, prediction_datetime):
        """Return the predicted value of the metric for the prediction_datetime."""
        nearest_index = self.predicted_df.index.get_indexer(
            [prediction_datetime], method="nearest"
        )[0]
        return self.predicted_df.iloc[[nearest_index]]

