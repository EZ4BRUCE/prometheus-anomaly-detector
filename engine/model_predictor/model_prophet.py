"""doctsring for packages."""

from datetime import datetime
from prometheus_api_client import PrometheusConnect
from prophet import Prophet
from prometheus_api_client import Metric
from engine.model_predictor.model_predictor import SeriesPredictor
import threading
from prometheus_client import Gauge
from prometheus_api_client.utils import parse_timedelta
import pandas as pd


class ProphetPredictor(SeriesPredictor):
    """docstring for Predictor."""

    model_name = "prophet"
    model_description = "Forecasted value from Prophet model"

    def __init__(
        self,
        logger,
        metric,
        series_hash,
        prometheus_url,
        gauge_metric: Gauge,
        rolling_data_window_size: str = "10d",
        future_offset: str = None,
    ):
        """Initialize the Metric object."""
        self.logger = logger
        oldest_data_datetime = parse_timedelta("now", rolling_data_window_size)
        self.metric = Metric(metric, oldest_data_datetime)
        self.series_hash = series_hash
        self.prometheus_client = PrometheusConnect(
            url=prometheus_url,
            disable_ssl=True,
        )
        self.future_offset = future_offset
        self.model = None
        self.predicted_df = None
        self.last_retrain_time = None
        self.gauge_metric = gauge_metric
        self.lock = threading.Lock()

    def get_series_hash(self):
        return self.series_hash

    def get_model_name(self):
        return self.model_name

    def get_model_description(self):
        return self.model_description

    def get_last_retrain_time(self):
        return self.last_retrain_time

    def train(self, metric_data=None, prediction_duration=15):
        """Train the Prophet model and store the predictions in predicted_df."""
        prediction_freq = "1min"  # Use lowercase 'min'

        # 将 future_offset 转换为分钟数
        if self.future_offset:
            future_offset_minutes = (
                pd.to_timedelta(self.future_offset).total_seconds() / 60
            )
            prediction_duration = prediction_duration + int(future_offset_minutes)

        # convert incoming metric to Metric Object
        if metric_data:
            # because the rolling_data_window_size is set, this df should not bloat
            self.metric += Metric(metric_data)

        # Don't really need to store the model, as prophet models are not retrainable
        # But storing it as an example for other models that can be retrained
        self.model = Prophet(
            daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True
        )

        self.logger.info(
            "[%s] Metric %s series %s training data range: %s - %s",
            "prophet",
            self.metric.metric_name,
            self.metric.label_config,
            self.metric.start_time,
            self.metric.end_time,
        )

        self.model.fit(self.metric.metric_values)
        future = self.model.make_future_dataframe(
            periods=int(prediction_duration),
            freq=prediction_freq,  # Updated frequency
            include_history=False,
        )
        forecast = self.model.predict(future)
        forecast["timestamp"] = forecast["ds"]
        forecast = forecast[["timestamp", "yhat", "yhat_lower", "yhat_upper"]]
        forecast = forecast.set_index("timestamp")
        new_predicted_df = forecast
        self.logger.debug(forecast)

        # 使用锁来更新 self.predicted_df
        with self.lock:
            self.predicted_df = new_predicted_df
            self.last_retrain_time = datetime.now()

    def predict_value(self, prediction_datetime):
        """Return the predicted value of the metric for the prediction_datetime."""
        # 读取 self.predicted_df 时使用锁
        with self.lock:
            nearest_index = self.predicted_df.index.get_indexer(
                [prediction_datetime], method="nearest"
            )[0]
            return self.predicted_df.iloc[[nearest_index]]

    def predict(self, time: datetime) -> bool:
        # get the current metric value so that it can be compared with the predicted values


        prediction = self.predict_value(time)

        # Check for all the columns available in the prediction
        # and publish the values for each of them
        for column_name in list(prediction.columns):

            public_labels_perdicted = {
                **self.metric.label_config,
                "value_type": column_name,
                "model_name": self.model_name,
                "metric_type": "anomaly-detection",
                "origin_metric_name": self.metric.metric_name,
            }

            if self.future_offset is not None:
                public_labels_perdicted["future_offset"] = self.future_offset

            self.gauge_metric.labels(**public_labels_perdicted).set(
                prediction[column_name].iloc[0]
            )

        # Calculate for an anomaly (can be different for different models)

        if self.future_offset is None:
            current_metric_data = self.prometheus_client.get_current_metric_value(
                self.metric.metric_name,
                self.metric.label_config,
            )

            if len(current_metric_data) == 0:
                return False

            current_metric_value = Metric(current_metric_data[0])
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
                "origin_metric_name": self.metric.metric_name,
            }

            # create a new time series that has value_type=anomaly
            # this value is 1 if an anomaly is found 0 if not
            self.gauge_metric.labels(**public_labels_anomaly).set(anomaly)
        return True
