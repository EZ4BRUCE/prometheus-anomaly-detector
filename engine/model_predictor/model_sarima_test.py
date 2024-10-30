import logging
import pandas as pd
import numpy as np
from prometheus_api_client import Metric
from statsmodels.tsa.statespace.sarimax import SARIMAX
from engine.model_predictor.model_predictor import SeriesPredictor
from prometheus_client import Gauge
from prometheus_api_client import PrometheusConnect
import threading
from datetime import datetime
class SarimaPredictor(SeriesPredictor):
    """SARIMA model for time series forecasting."""
    logger = None
    model_name = "sarima"
    model_description = "Forecast value based on SARIMA model"
    model = None
    predicted_df = None
    metric = None
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    series_hash = None
    last_retrain_time = None
    prometheus_client = None
    gauge_metric: Gauge = None
    lock = threading.Lock()

    def __init__(self, logger, metric, series_hash, prometheus_url, gauge_metric: Gauge, rolling_data_window_size="10d", order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """Initialize the SARIMA model."""
        oldest_data_datetime = parse_timedelta("now", rolling_data_window_size)
        self.metric = Metric(metric, oldest_data_datetime)
        self.logger = logger
        self.series_hash = series_hash
        self.prometheus_client = PrometheusConnect(
            url=prometheus_url,
            disable_ssl=True,
        )
        self.gauge_metric = gauge_metric
        self.order = order
        self.seasonal_order = seasonal_order

    def get_series_hash(self):
        return self.series_hash

    def get_model_name(self):
        return self.model_name

    def get_model_description(self):
        return self.model_description

    def get_last_retrain_time(self):
        return self.last_retrain_time


    def train(self, metric_data=None, prediction_duration=15):
        """Train the SARIMA model and store the predictions in pandas dataframe."""
        if metric_data:
            self.metric += Metric(metric_data)

        data = self.metric.metric_values
        vals = np.array(data["y"].tolist())

        self.logger.debug("Training data start time: %s", self.metric.start_time)
        self.logger.debug("Training data end time: %s", self.metric.end_time)
        self.logger.debug("Begin training")

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

        with self.lock:
            self.predicted_df = forecast
            self.last_retrain_time = datetime.now()

    def predict_value(self, prediction_datetime):
        """Return the predicted value of the metric for the prediction_datetime."""
        with self.lock:
            nearest_index = self.predicted_df.index.get_indexer(
                [prediction_datetime], method="nearest"
            )[0]
            return self.predicted_df.iloc[[nearest_index]]

    def predict(self, time: datetime):

        self.logger.info(
            f"model {self.model_name} predicting value for series {self.metric.label_config} at {time}"
        )

        # get the current metric value so that it can be compared with the predicted values
        current_metric_value = Metric(
            self.prometheus_client.get_current_metric_value(
                self.metric.metric_name,
                self.metric.label_config,
            )[0]
        )

        prediction = self.predict_value(time)

        # Check for all the columns available in the prediction
        # and publish the values for each of them
        for column_name in list(prediction.columns):

            public_labels_perdicted = {
                **self.metric.label_config,
                "value_type": column_name,
                "model_name": self.model_name,
                "metric_type": "anomaly-detection",
            }

            self.gauge_metric.labels(**public_labels_perdicted).set(
                prediction[column_name].iloc[0]
            )

            # Calculate for an anomaly (can be different for different models)
            anomaly = 1
            if (
                current_metric_value.metric_values["y"].iloc[0]
                < prediction["yhat_upper"].iloc[0]
            ) and (
                current_metric_value.metric_values["y"].iloc[0]
                > prediction["yhat_lower"].iloc[0]
            ):
                anomaly = 0

            public_labels_anomaly = {
                **self.metric.label_config,
                "value_type": "anomaly",
                "model_name": self.model_name,
                "metric_type": "anomaly-detection",
            }

            # create a new time series that has value_type=anomaly
            # this value is 1 if an anomaly is found 0 if not
            self.gauge_metric.labels(**public_labels_anomaly).set(anomaly)
