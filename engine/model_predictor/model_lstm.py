"""doctsring for packages."""

import logging
from prometheus_api_client import Metric
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from engine.model_predictor.model_predictor import SeriesPredictor
import threading
from prometheus_client import Gauge
from prometheus_api_client import PrometheusConnect
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime


class LstmPredictor(SeriesPredictor):
    """docstring for Predictor."""

    logger = None
    model_name = "lstm"
    model_description = "Forecasted value from Lstm model"
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
        number_of_feature=10,
        validation_ratio=0.2,
        parameter_tuning=True,
    ):
        """Initialize the Metric object."""
        self.logger = logger
        self.metric = Metric(metric, rolling_data_window_size)
        self.series_hash = series_hash
        self.prometheus_client = PrometheusConnect(
            url=prometheus_url,
            disable_ssl=True,
        )  
        self.gauge_metric = gauge_metric
        self.number_of_features = number_of_feature
        self.scalar = MinMaxScaler(feature_range=(0, 1))
        self.parameter_tuning = parameter_tuning
        self.validation_ratio = validation_ratio

    def get_series_hash(self):
        return self.series_hash

    def get_model_name(self):
        return self.model_name

    def get_model_description(self):
        return self.model_description

    def get_last_retrain_time(self):
        return self.last_retrain_time


    def prepare_data(self, data):
        """Prepare the data for LSTM."""
        # 检查并清理数据中的 NaN 或 Inf

        train_x = np.array(data[:, 1])[np.newaxis, :].T

        for i in range(self.number_of_features):
            train_x = np.concatenate(
                (train_x, np.roll(data[:, 1], -i)[np.newaxis, :].T), axis=1
            )

        train_x = train_x[
            : train_x.shape[0] - self.number_of_features, : self.number_of_features
        ]

        train_yt = np.roll(data[:, 1], -self.number_of_features + 1)
        train_y = np.roll(data[:, 1], -self.number_of_features)
        train_y = train_y - train_yt
        train_y = train_y[: train_y.shape[0] - self.number_of_features]

        train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
        return train_x, train_y

    def get_model(self, lstm_cell_count, dense_cell_count):
        """Build the model."""
        model = Sequential()
        # Use Input layer to define the input shape
        model.add(Input(shape=(1, self.number_of_features)))
        model.add(LSTM(64, return_sequences=True))
        model.add(LSTM(lstm_cell_count))
        model.add(Dense(dense_cell_count))
        model.add(Dense(1))
        return model

    def train(self, metric_data=None, prediction_duration=15):
        """Train the model."""
        if metric_data:
            # because the rolling_data_window_size is set, this df should not bloat
            self.metric += Metric(metric_data)

        # normalising
        metric_values_np = self.metric.metric_values.values
        scaled_np_arr = self.scalar.fit_transform(metric_values_np[:, 1].reshape(-1, 1))
        metric_values_np[:, 1] = scaled_np_arr.flatten()

        if self.parameter_tuning:
            x, y = self.prepare_data(metric_values_np)
            lstm_cells = [2**i for i in range(5, 8)]
            dense_cells = [2**i for i in range(5, 8)]
            loss = np.inf
            lstm_cell_count = 0
            dense_cell_count = 0
            for lstm_cell_count_ in lstm_cells:
                for dense_cell_count_ in dense_cells:
                    model = self.get_model(lstm_cell_count_, dense_cell_count_)
                    model.compile(loss="mean_squared_error", optimizer="adam")
                    history = model.fit(
                        np.asarray(x).astype(np.float32),
                        np.asarray(y).astype(np.float32),
                        epochs=50,
                        batch_size=512,
                        verbose=0,
                        validation_split=self.validation_ratio,
                    )
                    val_loss = history.history["val_loss"]
                    loss_ = min(val_loss)
                    if loss > loss_:
                        lstm_cell_count = lstm_cell_count_
                        dense_cell_count = dense_cell_count_
                        loss = loss_
            self.lstm_cell_count = lstm_cell_count
            self.dense_cell_count = dense_cell_count
            self.parameter_tuning = False

        model = self.get_model(self.lstm_cell_count, self.dense_cell_count)
        self.logger.info(
            "training data range: %s - %s", self.metric.start_time, self.metric.end_time
        )
        # _LOGGER.info("training data end time: %s", self.metric.end_time)
        self.logger.debug("begin training")
        data_x, data_y = self.prepare_data(metric_values_np)
        self.logger.debug(data_x.shape)
        model.compile(loss="mean_squared_error", optimizer="adam")
        model.fit(
            np.asarray(data_x).astype(np.float32),
            np.asarray(data_y).astype(np.float32),
            epochs=50,
            batch_size=512,
        )
        data_test = np.asarray(metric_values_np[-self.number_of_features :, 1]).astype(
            np.float32
        )
        forecast_values = []
        prev_value = data_test[-1]
        for i in range(int(prediction_duration)):
            prediction = model.predict(
                data_test.reshape(1, 1, self.number_of_features)
            ).flatten()[0]
            curr_pred_value = data_test[-1] + prediction
            scaled_final_value = self.scalar.inverse_transform(
                curr_pred_value.reshape(1, -1)
            ).flatten()[0]
            forecast_values.append(scaled_final_value)
            data_test = np.roll(data_test, -1)
            data_test[-1] = curr_pred_value
            prev_value = data_test[-1]

        dataframe_cols = {"yhat": np.array(forecast_values)}

        upper_bound = np.array(
            [
                (forecast_values[i] + (np.std(forecast_values[:i]) * 2))
                for i in range(len(forecast_values))
            ]
        )
        upper_bound[0] = np.mean(
            forecast_values[0]
        )  # to account for no std of a single value
        lower_bound = np.array(
            [
                (forecast_values[i] - (np.std(forecast_values[:i]) * 2))
                for i in range(len(forecast_values))
            ]
        )
        lower_bound[0] = np.mean(
            forecast_values[0]
        )  # to account for no std of a single value
        dataframe_cols["yhat_upper"] = upper_bound
        dataframe_cols["yhat_lower"] = lower_bound

        data = self.metric.metric_values
        maximum_time = max(data["ds"])
        dataframe_cols["timestamp"] = pd.date_range(
            maximum_time, periods=len(forecast_values), freq="min"
        )

        forecast = pd.DataFrame(data=dataframe_cols)
        forecast = forecast.set_index("timestamp")

        with self.lock:
            self.predicted_df = forecast
            self.last_retrain_time = datetime.now()
        self.logger.debug(forecast)

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