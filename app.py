"""docstring for packages."""

import time
import os
import logging
from datetime import datetime
from multiprocessing import Pool, Process, Queue
from multiprocessing import cpu_count
from functools import partial
from queue import Empty as EmptyQueueException
import tornado.ioloop
import tornado.web
from prometheus_client import Gauge, generate_latest, REGISTRY
from prometheus_api_client import PrometheusConnect, Metric
from prometheus_api_client.utils import parse_datetime, parse_timedelta
from configuration import Configuration
import model_prophet
import model_fourier
import model_lstm
import schedule
import asyncio
import json
import model_sarima_test

# Set up logging
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()

# Create a formatter that includes the line number
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]"
)

# Set the formatter for the handler
console_handler.setFormatter(formatter)

# Add the handler to the logger
_LOGGER.addHandler(console_handler)

METRICS_LIST = Configuration.metrics_list

# 初始化为空列表
PREDICTOR_MODEL_LIST = list()

print(f"Prometheus URL: {Configuration.prometheus_url}")

pc = PrometheusConnect(
    url=Configuration.prometheus_url,
    headers=Configuration.prom_connect_headers,
    disable_ssl=True,
)

# 去除初始化添加指标的训练
# for metric in METRICS_LIST:
#     # Initialize a predictor for all metrics first
#     metric_init = pc.get_current_metric_value(metric_name=metric)

#     for unique_metric in metric_init:
#         # 在这里定义选择不同的模型，可以根据 label 来
#         PREDICTOR_MODEL_LIST.append(
#             model.MetricPredictor(
#                 unique_metric,
#                 rolling_data_window_size=Configuration.rolling_training_window_size,
#             )
#         )

# A gauge set for the predicted values
GAUGE_DICT = dict()
# for predictor in PREDICTOR_MODEL_LIST:
#     unique_metric = predictor.metric
#     label_list = list(unique_metric.label_config.keys())
#     label_list.append("value_type")
#     if unique_metric.metric_name not in GAUGE_DICT:
#         GAUGE_DICT[unique_metric.metric_name] = Gauge(
#             unique_metric.metric_name + "_" + predictor.model_name,
#             predictor.model_description,
#             label_list,
#         )


class MainHandler(tornado.web.RequestHandler):
    """Tornado web request handler."""

    def initialize(self, data_queue):
        """Check if new predicted values are available in the queue before the get request."""
        try:
            model_list = data_queue.get_nowait()
            self.settings["model_list"] = model_list
        except EmptyQueueException:
            pass

    async def get(self):
        """Fetch and publish metric values asynchronously."""
        # update metric value on every request and publish the metric
        for predictor_model in self.settings["model_list"]:
            # get the current metric value so that it can be compared with the
            # predicted values
            current_metric_value = Metric(
                pc.get_current_metric_value(
                    metric_name=predictor_model.metric.metric_name,
                    label_config=predictor_model.metric.label_config,
                )[0]
            )

            metric_name = predictor_model.metric.metric_name
            prediction = predictor_model.predict_value(datetime.now())

            # Check for all the columns available in the prediction
            # and publish the values for each of them
            for column_name in list(prediction.columns):
                GAUGE_DICT[metric_name].labels(
                    **predictor_model.metric.label_config,
                    value_type=column_name,
                    model_name=predictor_model.model_name,
                    metric_type="anomaly-detection",
                ).set(prediction[column_name][0])

            # Calculate for an anomaly (can be different for different models)
            anomaly = 1
            if (
                current_metric_value.metric_values["y"][0] < prediction["yhat_upper"][0]
            ) and (
                current_metric_value.metric_values["y"][0] > prediction["yhat_lower"][0]
            ):
                anomaly = 0
            # create a new time series that has value_type=anomaly
            # this value is 1 if an anomaly is found 0 if not
            GAUGE_DICT[metric_name].labels(
                **predictor_model.metric.label_config,
                value_type="anomaly",
                model_name=predictor_model.model_name,
                metric_type="anomaly-detection",
            ).set(anomaly)

        self.write(generate_latest(REGISTRY).decode("utf-8"))
        self.set_header("Content-Type", "text; charset=utf-8")


class AddMetricHandler(tornado.web.RequestHandler):
    """Handler to add new metrics for training."""

    def initialize(self, data_queue):
        """Initialize with data queue."""
        self.data_queue = data_queue

    async def post(self):
        """Add a new metric and trigger model training."""
        try:
            # Parse JSON body
            data = json.loads(self.request.body)
            new_metric = data.get("metric")
            model_name = data.get("model")
            window_size = data.get("window_size")

            time_window = parse_timedelta("now", window_size)

            _LOGGER.info(
                f"Received new metric for training: {new_metric}, model: {model_name}"
            )

            # Get current metric value
            metric_init = pc.get_current_metric_value(metric_name=new_metric)

            for predictor in PREDICTOR_MODEL_LIST:
                _LOGGER.info(
                    f"current {predictor.metric.metric_name} predictor: {predictor.model_name} {predictor.metric.label_config}"
                )

            # List to store newly added predictors
            new_predictors = []

            for m in metric_init:
                # Check if the unique_metric already exists in PREDICTOR_MODEL_LIST
                _LOGGER.info(f"get unique_metric: {m}")

                label_config_with_matric_name = m["metric"]
                metric_name = label_config_with_matric_name["__name__"]

                is_initial_run = not any(
                    predictor.metric.metric_name == metric_name
                    and predictor.label_config_with_matric_name
                    == label_config_with_matric_name
                    for predictor in PREDICTOR_MODEL_LIST
                )

                # If it doesn't exist, add a new predictor to the list
                if is_initial_run:
                    if time_window == None:
                        time_window = Configuration.rolling_training_window_size
                    new_predictor = self.new_model_predictor(
                        m, label_config_with_matric_name, model_name, time_window
                    )
                    new_predictors.append(new_predictor)
                    PREDICTOR_MODEL_LIST.append(new_predictor)

                    unique_metric = new_predictor.metric

                    # Update GAUGE_DICT
                    label_list = list(unique_metric.label_config.keys())
                    label_list.append("value_type")
                    label_list.append("model_name")
                    label_list.append("metric_type")
                    if unique_metric.metric_name not in GAUGE_DICT:
                        GAUGE_DICT[unique_metric.metric_name] = Gauge(
                            unique_metric.metric_name + "_" + new_predictor.model_name,
                            new_predictor.model_description,
                            labelnames=label_list,
                        )

            # Train only the newly added predictors
            if new_predictors:
                # Schedule the training as a background task
                asyncio.create_task(
                    train_model_async(
                        new_predictors, initial_run=True, data_queue=self.data_queue
                    )
                )
                self.write(
                    {
                        "status": "success",
                        "message": f"Metric [{new_metric}] added and training started.",
                    }
                )
            else:
                _LOGGER.info(
                    f"Metric [{new_metric}] predictor already exists. Skipping training."
                )
                self.write(
                    {
                        "status": "success",
                        "message": f"Metric [{new_metric}]'s predictor already exists. Skipping training.",
                    }
                )

        except json.JSONDecodeError:
            self.set_status(400)
            self.write({"status": "error", "message": "Invalid JSON"})
        except Exception as e:
            _LOGGER.error(f"Error adding new metric: {str(e)}")
            self.write({"status": "error", "message": str(e)})

    def new_model_predictor(
        self,
        unique_metric,
        label_config_with_matric_name,
        model_name,
        rolling_data_window_size,
    ):
        if model_name == "prophet":
            return model_prophet.MetricPredictor(
                unique_metric,
                label_config_with_matric_name,
                rolling_data_window_size=rolling_data_window_size,
            )
        elif model_name == "fourier":
            return model_fourier.MetricPredictor(
                unique_metric,
                label_config_with_matric_name,
                rolling_data_window_size=rolling_data_window_size,
            )
        elif model_name == "lstm":
            return model_lstm.MetricPredictor(
                unique_metric,
                label_config_with_matric_name,
                rolling_data_window_size=rolling_data_window_size,
            )
        elif model_name == "sarima":
            # still in test
            return model_sarima_test.MetricPredictor(
                unique_metric,
                label_config_with_matric_name,
                rolling_data_window_size=rolling_data_window_size,
            )
        else:
            raise ValueError(f"Invalid model name: {model_name}")


def make_app(data_queue):
    """Initialize the tornado web app."""
    _LOGGER.info("Initializing Tornado Web App")
    return tornado.web.Application(
        [
            (r"/metrics", MainHandler, dict(data_queue=data_queue)),
            (r"/", MainHandler, dict(data_queue=data_queue)),
            (
                r"/add_metric",
                AddMetricHandler,
                dict(data_queue=data_queue),
            ),  # New API endpoint
        ],
        settings={"data_queue": data_queue},  # Add data_queue to settings
    )


async def train_individual_model_async(predictor_model, initial_run):
    """Asynchronously train an individual model."""
    metric_to_predict = predictor_model.metric
    pc = PrometheusConnect(
        url=Configuration.prometheus_url,
        headers=Configuration.prom_connect_headers,
        disable_ssl=True,
    )

    data_start_time = datetime.now() - Configuration.metric_chunk_size
    if initial_run:
        data_start_time = datetime.now() - Configuration.rolling_training_window_size

    # Download new metric data from prometheus
    new_metric_data = pc.get_metric_range_data(
        metric_name=metric_to_predict.metric_name,
        label_config=metric_to_predict.label_config,
        start_time=data_start_time,
        end_time=datetime.now(),
    )[0]

    # Train the new model
    start_time = datetime.now()
    predictor_model.train(new_metric_data, Configuration.retraining_interval_minutes)

    _LOGGER.info(
        "Total Training time taken = %s, for metric: %s %s",
        str(datetime.now() - start_time),
        metric_to_predict.metric_name,
        metric_to_predict.label_config,
    )
    return predictor_model


async def train_model_async(predictors, initial_run=False, data_queue=None):
    """Asynchronously train the machine learning models."""
    if not predictors:
        _LOGGER.info("No new metrics to train. Skipping training.")
        return

    _LOGGER.info(f"Training models asynchronously with asyncio")

    # Create asynchronous tasks for each predictor
    tasks = [
        train_individual_model_async(predictor, initial_run) for predictor in predictors
    ]

    # Run all tasks concurrently
    result = await asyncio.gather(*tasks)

    # Update global PREDICTOR_MODEL_LIST
    for predictor in result:
        if predictor not in PREDICTOR_MODEL_LIST:
            PREDICTOR_MODEL_LIST.append(predictor)

    data_queue.put(PREDICTOR_MODEL_LIST)


if __name__ == "__main__":
    # Queue to share data between the tornado server and the model training
    predicted_model_queue = Queue()

    # 不进行初始模型训练，改为通过 API 添加指标训练
    # train_model(initial_run=True, data_queue=predicted_model_queue)

    # Set up the tornado web app
    app = make_app(predicted_model_queue)
    app.listen(8789)
    server_process = Process(target=tornado.ioloop.IOLoop.instance().start)
    # Start up the server to expose the metrics.
    server_process.start()

    # Schedule the model training
    schedule.every(Configuration.retraining_interval_minutes).minutes.do(
        train_model_async, initial_run=False, data_queue=predicted_model_queue
    )
    _LOGGER.info(
        "Will retrain model every %s minutes", Configuration.retraining_interval_minutes
    )

    while True:
        schedule.run_pending()
        time.sleep(1)

    # join the server process in case the main process ends
    server_process.join()
