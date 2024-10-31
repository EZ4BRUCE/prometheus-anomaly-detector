"""doctsring for packages."""

from datetime import datetime
from prometheus_api_client import PrometheusConnect
from prophet import Prophet
from prometheus_api_client import Metric
from engine.model_predictor.model_predictor import SeriesPredictor
import threading
from prometheus_client import Gauge
from prometheus_api_client.utils import parse_timedelta


class ProphetPredictor(SeriesPredictor):
    """docstring for Predictor."""

    logger = None
    model_name = "prophet"
    model_description = "Forecasted value from Prophet model"
    model = None
    predicted_df = None
    metric = None
    series_hash = None
    last_retrain_time = None
    prometheus_client = None
    gauge_metric: Gauge = None
    lock = threading.Lock()

    def __init__(
        self,
        logger,
        metric,
        series_hash,
        prometheus_url,
        gauge_metric: Gauge,
        rolling_data_window_size="10d",
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
        self.gauge_metric = gauge_metric

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
            "[%s] training data range: %s - %s",
            "prophet",
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
        current_metric_data = self.prometheus_client.get_current_metric_value(
            self.metric.metric_name,
            self.metric.label_config,
        )

        if len(current_metric_data) == 0:
            return False

        current_metric_value = Metric(current_metric_data[0])

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
                "origin_metric_name": self.metric.metric_name,
            }

            # create a new time series that has value_type=anomaly
            # this value is 1 if an anomaly is found 0 if not
            self.gauge_metric.labels(**public_labels_anomaly).set(anomaly)
            return True
