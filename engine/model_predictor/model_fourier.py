"""docstring for installed packages."""
from datetime import datetime
import pandas as pd
import numpy as np
from prometheus_api_client import Metric
from numpy import fft
from engine.model_predictor.model_predictor import SeriesPredictor
import threading
from prometheus_client import Gauge
from prometheus_api_client import PrometheusConnect

class FourierPredictor(SeriesPredictor):
    """docstring for Predictor."""
    logger = None
    model_name = "fourier"
    model_description = "Forecast value based on fourier analysis"
    model = None
    predicted_df = None

    metric = None
    series_hash = None
    last_retrain_time = None
    prometheus_client = None
    gauge_metric: Gauge = None
    lock = threading.Lock()

    def __init__(self, logger, metric, series_hash, prometheus_url, gauge_metric: Gauge, rolling_data_window_size="10d"):
        """Initialize metric object."""
        self.metric = Metric(metric, rolling_data_window_size)
        self.logger = logger
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

    def fourier_extrapolation(self, input_series, n_predict, n_harmonics):
        """Perform the Fourier extrapolation on time series data."""
        n = input_series.size
        t = np.arange(0, n)
        p = np.polyfit(t, input_series, 1)
        input_no_trend = input_series - p[0] * t
        frequency_domain = fft.fft(input_no_trend)
        frequencies = fft.fftfreq(n)
        indexes = np.arange(n).tolist()
        indexes.sort(key=lambda i: np.absolute(frequencies[i]))

        time_steps = np.arange(0, n + n_predict)
        restored_signal = np.zeros(time_steps.size)

        for i in indexes[: 1 + n_harmonics * 2]:
            amplitude = np.absolute(frequency_domain[i]) / n
            phase = np.angle(frequency_domain[i])
            restored_signal += amplitude * np.cos(
                2 * np.pi * frequencies[i] * time_steps + phase
            )

        restored_signal = restored_signal + p[0] * time_steps
        return restored_signal[n:]

    def train(self, metric_data=None, prediction_duration=15):
        """Train the Fourier model and store the predictions in pandas dataframe."""
        prediction_range = prediction_duration
        # convert incoming metric to Metric Object
        if metric_data:
            # because the rolling_data_window_size is set, this df should not bloat
            self.metric += Metric(metric_data)

        data = self.metric.metric_values
        vals = np.array(data["y"].tolist())

        self.logger.debug("training data start time: %s", self.metric.start_time)
        self.logger.debug("training data end time: %s", self.metric.end_time)
        self.logger.debug("begin training")

        forecast_values = self.fourier_extrapolation(
            vals, prediction_range, 1
        )  # int(len(vals)/3))
        dataframe_cols = {}
        dataframe_cols["yhat"] = np.array(forecast_values)

        # find most recent timestamp from original data and extrapolate new timestamps
        self.logger.debug("Creating Dummy Timestamps.....")
        maximum_time = max(data["ds"])
        dataframe_cols["timestamp"] = pd.date_range(
            maximum_time, periods=len(forecast_values), freq="min"
        )

        # create dummy upper and lower bounds
        self.logger.debug("Computing Bounds .... ")

        upper_bound = np.array(
            [
                (
                    np.ma.average(
                        forecast_values[:i],
                        weights=np.linspace(0, 1, num=len(forecast_values[:i])),
                    )
                    + (np.std(forecast_values[:i]) * 2)
                )
                for i in range(len(forecast_values))
            ]
        )
        upper_bound[0] = np.mean(
            forecast_values[0]
        )  # to account for no std of a single value
        lower_bound = np.array(
            [
                (
                    np.ma.average(
                        forecast_values[:i],
                        weights=np.linspace(0, 1, num=len(forecast_values[:i])),
                    )
                    - (np.std(forecast_values[:i]) * 2)
                )
                for i in range(len(forecast_values))
            ]
        )
        lower_bound[0] = np.mean(
            forecast_values[0]
        )  # to account for no std of a single value
        dataframe_cols["yhat_upper"] = upper_bound
        dataframe_cols["yhat_lower"] = lower_bound

        # create series and index into predictions_dict
        self.logger.debug("Formatting Forecast to Pandas ..... ")

        forecast = pd.DataFrame(data=dataframe_cols)
        forecast = forecast.set_index("timestamp")

        with self.lock:
            self.predicted_df = forecast
            self.last_retrain_time = datetime.now()

    def predict_value(self, prediction_datetime):
        """Return the predicted value of the metric for the prediction_datetime."""
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
