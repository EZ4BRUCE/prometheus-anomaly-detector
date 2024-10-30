import time
import logging
import threading
import schedule
import asyncio

from engine.model_predictor.model_predictor import SeriesPredictor


from engine.model_predictor.model_prophet import ProphetPredictor
from engine.model_predictor.model_fourier import FourierPredictor
from engine.model_predictor.model_lstm import LstmPredictor
from engine.model_predictor.model_sarima_test import SarimaPredictor

from prometheus_client import Gauge
from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_timedelta
from copy import deepcopy
from datetime import datetime, timedelta
from prometheus_client import REGISTRY

CONST_METRIC_NAME_LABEL_KEY = "__name__"


class MetricAnalyzer:
    logger: logging.Logger = None
    cluster_mode: bool = False
    # must be a metric promql(or a reconrding rule metric name), not a promql
    # should be like: `container_memory_usage_bytes{namespace="default", pod="pod-name"}`
    metric_promql: str = None
    prometheus_url: str = None
    prometheus_client: PrometheusConnect = None
    model_name: str = None
    rolling_data_window_size: str = None
    # eg: 120m
    retraining_interval_minutes: int = 120
    sync_new_series_interval_seconds: int = 300

    # metric name, like: `container_memory_usage_bytes`
    metric_name: str = None
    # labels without __name__
    series_label_keys: set[str] = None
    gauge_metric: Gauge = None
    # label key-value hash -> ModelPredictor
    series_predictors: dict[str, SeriesPredictor] = {}
    predictor_dict_lock: threading.Lock = None

    background_thread: threading.Thread = None
    stop_event: threading.Event = None

    is_stopped: bool = False

    def __init__(
        self,
        logger,
        cluster_mode,
        metric_promql,
        prometheus_url,
        model_name,
        rolling_data_window_size,
        retraining_interval_minutes,
        sync_new_series_interval_seconds,
    ):
        self.logger = logger
        self.cluster_mode = cluster_mode
        self.metric_promql = metric_promql
        self.prometheus_url = prometheus_url
        self.prometheus_client = PrometheusConnect(
            url=prometheus_url,
            disable_ssl=True,
        )
        self.model_name = model_name
        self.rolling_data_window_size = rolling_data_window_size
        self.retraining_interval_minutes = retraining_interval_minutes
        self.predictor_dict_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.sync_new_series_interval_seconds = sync_new_series_interval_seconds

    def set_rolling_data_window_size(self, rolling_data_window_size: str):
        self.rolling_data_window_size = parse_timedelta("now", rolling_data_window_size)

    def set_retraining_interval_minutes(self, retraining_interval_minutes: int):
        self.retraining_interval_minutes = retraining_interval_minutes

    def same_label_keys(self, keys):
        for k in keys:
            if k not in self.series_label_keys:
                return False
        return True

    async def predict_all_series_values(self):

        now = datetime.now()

        with self.predictor_dict_lock:
            if len(self.series_predictors) == 0:
                self.logger.warning(
                    "[%s] %s(id: %s) no series to predict",
                    "analyzer",
                    self.metric_promql,
                    id(self),
                )
                return

            for hash, predictor in self.series_predictors.items():
                predictor.predict(now)

            self.logger.info(
                "[%s] all series values updated in gauge metrics for %s(id: %s)",
                "analyzer",
                self.metric_promql,
                id(self),
            )

    async def check_and_retrain_predictors(self):
        """Asynchronously retrain the predictors that need updating."""
        current_time = datetime.now()

        # 过滤出需要重新训练的预测器
        with self.predictor_dict_lock:
            retrain_predictors = [
                predictor
                for predictor in self.series_predictors.values()
                if (current_time - predictor.last_retrain_time).total_seconds() / 60
                >= self.retraining_interval_minutes
            ]

        if not retrain_predictors:
            self.logger.info(
                "[%s] No predictors need retraining at this time.", "analyzer"
            )
            return

        self.logger.info(
            "[%s] Retraining %s predictors.", "analyzer", len(retrain_predictors)
        )

        # 直接将 retrain_predictors 列表传递给 train_model_async
        await self.train_model_async(retrain_predictors, initial_run=False)

        # 更新 last_retrain_time
        for predictor in retrain_predictors:
            predictor.last_retrain_time = current_time

    def series_data_ready(self, metric_name, labels, time_range) -> bool:
        data_start_time = datetime.now() - parse_timedelta(
            "now", time_range
        )
        data_end_time = data_start_time + timedelta(seconds=1200)
        new_series_data = self.prometheus_client.get_metric_range_data(
            metric_name=metric_name,
            label_config=labels,
            start_time=data_start_time,
            end_time=data_end_time, 
        )
        return len(new_series_data) > 0

    # 1. api call
    # 2. auto sync series_predictors
    def sync_new_series(self):
        try:
            # Get current metric value
            current_series = self.prometheus_client.get_current_metric_value(
                metric_name=self.metric_promql
            )

            if len(current_series) == 0:
                self.logger.info("no series to add")
                return

            self.logger.info(
                "[%s] got %s series total", "analyzer", len(current_series)
            )

            new_predictors = []

            if self.series_label_keys is None:
                labels = deepcopy(current_series[0]["metric"])
                if CONST_METRIC_NAME_LABEL_KEY not in labels:
                    raise ValueError(
                        f"metric {self.metric_promql} has no {CONST_METRIC_NAME_LABEL_KEY}"
                    )
                self.metric_name = labels[CONST_METRIC_NAME_LABEL_KEY]
                del labels[CONST_METRIC_NAME_LABEL_KEY]
                self.series_label_keys = labels.keys()

            # Update GAUGE_DICT
            if self.metric_name is not None and self.gauge_metric is None:
                publish_labels = list(self.series_label_keys)
                publish_labels.append("value_type")
                publish_labels.append("model_name")
                publish_labels.append("metric_type")
                publish_labels.append("origin_metric_name")
                self.gauge_metric = Gauge(
                    self.metric_name + "_" + self.model_name,
                    "Forecasted value by " + self.model_name,
                    labelnames=publish_labels,
                )

            for series in current_series:
                if CONST_METRIC_NAME_LABEL_KEY not in series["metric"]:
                    raise ValueError(
                        "[%s] metric %s has no %s",
                        "analyzer",
                        series["metric"],
                        CONST_METRIC_NAME_LABEL_KEY,
                    )

                metric_name = series["metric"][CONST_METRIC_NAME_LABEL_KEY]
                self.metric_name = metric_name

                labels = deepcopy(series["metric"])
                del labels[CONST_METRIC_NAME_LABEL_KEY]

                if not self.same_label_keys(labels.keys()):
                    self.logger.warning(
                        "[%s] label keys not match: %s", "analyzer", labels.keys()
                    )
                    continue
                if not self.series_data_ready(metric_name, labels, self.rolling_data_window_size):
                    self.logger.warning(
                        "[%s] data is not ready(%s) to train for metric: %s series: %s, skip training",
                        "analyzer",
                        self.rolling_data_window_size,
                        self.metric_promql,
                        series["metric"],
                    )
                    continue
                else:
                    self.logger.info(
                        "[%s] data is ready(%s) to train for metric: %s series: %s, start training",
                        "analyzer",
                        self.rolling_data_window_size,
                        self.metric_promql,
                        series["metric"],
                    )

                series_label_hash = hash(frozenset(labels.items()))
                with self.predictor_dict_lock:
                    if series_label_hash not in self.series_predictors:
                        self.logger.info("[%s] got new series: %s", "analyzer", series)
                        new_predictor = self.new_model_predictor(
                            series,
                            series_label_hash,
                            self.model_name,
                            self.prometheus_url,
                            self.gauge_metric,
                            self.rolling_data_window_size,
                        )
                        new_predictors.append(new_predictor)

            # Train only the newly added predictors
            if len(new_predictors) > 0:
                # Schedule the training as a background task
                self.logger.info(
                    "[%s] Training %s new series predictors for metric: %s",
                    "analyzer",
                    len(new_predictors),
                    self.metric_promql,
                )
                asyncio.run(self.train_model_async(new_predictors, initial_run=True))
            else:
                self.logger.info(
                    "[%s] Metric [%s] all series' predictors already exists. Skipping training.",
                    "analyzer",
                    self.metric_name,
                )
        except Exception as e:
            self.logger.error("[%s] Error syncing new series: %s", "analyzer", str(e))

    def new_model_predictor(
        self,
        series,
        series_hash,
        model_name,
        prometheus_url,
        gauge_metric: Gauge,
        rolling_data_window_size,
    ) -> SeriesPredictor:
        if model_name == "prophet":
            return ProphetPredictor(
                self.logger,
                series,
                series_hash,
                prometheus_url,
                gauge_metric,
                rolling_data_window_size=rolling_data_window_size,
            )
        elif model_name == "fourier":
            return FourierPredictor(
                self.logger,
                series,
                series_hash,
                prometheus_url,
                gauge_metric,
                rolling_data_window_size=rolling_data_window_size,
            )
        elif model_name == "lstm":
            return LstmPredictor(
                self.logger,
                series,
                series_hash,
                prometheus_url,
                gauge_metric,
                rolling_data_window_size=rolling_data_window_size,
            )
        elif model_name == "sarima":
            # still in test
            return SarimaPredictor(
                self.logger,
                series,
                series_hash,
                prometheus_url,
                gauge_metric,
                rolling_data_window_size=rolling_data_window_size,
            )
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    async def train_model_async(
        self, predictors: list[SeriesPredictor], initial_run=False
    ):
        """Asynchronously train the machine learning models."""
        if not predictors:
            self.logger.warning("[%s] No series to train. Skipping training.", "analyzer")
            return

        self.logger.info("[%s] Training models asynchronously with asyncio", "analyzer")

        # Create asynchronous tasks for each predictor
        tasks = [
            self.train_individual_model_async(predictor, initial_run)
            for predictor in predictors
        ]

        with self.predictor_dict_lock:
            if self.is_stopped:
                self.logger.info(
                    "[%s] promql analyzer for %s already stopped",
                    "analyzer",
                    self.metric_promql,
                )
                return

        # Run all tasks concurrently
        result = await asyncio.gather(*tasks)

        # Update global PREDICTOR_MODEL_LIST
        with self.predictor_dict_lock:
            if len(result) == 0:
                self.logger.info("[%s] no predictor trained", "analyzer")
                return

            for predictor in result:
                if predictor is not None:
                    self.series_predictors[predictor.get_series_hash()] = predictor

            self.logger.info(
                "[%s] %s predictors added", "analyzer", len(self.series_predictors)
            )

    async def train_individual_model_async(
        self, predictor_model: SeriesPredictor, initial_run: bool
    ):
        """Asynchronously train an individual model."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self.train_model, predictor_model, initial_run
            )
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            return None

    def train_model(
        self, predictor_model: SeriesPredictor, initial_run: bool
    ) -> SeriesPredictor:
        """Train the model in a separate thread."""
        series_to_predict = predictor_model.metric

        data_start_time = datetime.now() - parse_timedelta(
            "now", str(self.retraining_interval_minutes) + "m"
        )
        if initial_run:
            data_start_time = datetime.now() - parse_timedelta(
                "now", self.rolling_data_window_size
            )
        # Download new metric data from prometheus
        new_series_data = self.prometheus_client.get_metric_range_data(
            metric_name=series_to_predict.metric_name,
            label_config=series_to_predict.label_config,
            start_time=data_start_time,
            end_time=datetime.now(),
        )[0]

        # Train the new model
        start_time = datetime.now()
        # 预测区间更宽，预留 10 分钟（多10个数据点）
        predictor_model.train(new_series_data, self.retraining_interval_minutes + 10)

        self.logger.info(
            "Total Training time taken = %s, for metric: %s %s",
            str(datetime.now() - start_time),
            series_to_predict.metric_name,
            series_to_predict.label_config,
        )
        return predictor_model

    def run(self):
        """Run the retrain_predictors method at regular intervals."""
        if len(self.series_predictors) == 0:
            self.logger.info(
                "[%s] init promql analyzer for %s", "analyzer", self.metric_promql
            )
            self.sync_new_series()

        self.logger.info(
            "[%s] %s(id: %s) initial training for %d series done",
            "analyzer",
            self.metric_promql,
            id(self),
            len(self.series_predictors),
        )

        # Schedule retrain_predictors to run every retraining_interval_minutes
        schedule.every(90).seconds.do(
            lambda: asyncio.run(self.check_and_retrain_predictors())
        )

        self.logger.info(
            "[%s] Scheduled check predictors retrain schedule every 30 seconds.",
            "analyzer",
        )

        schedule.every(self.sync_new_series_interval_seconds).seconds.do(
            lambda: asyncio.run(self.resync_series())
        )

        self.logger.info(
            "[%s] Scheduled sync_new_series every %s seconds.",
            "analyzer",
            self.sync_new_series_interval_seconds,
        )

        while not self.stop_event.is_set():
            schedule.run_pending()
            time.sleep(1)

    async def resync_series(self):
        self.logger.info(
            "[%s] resync series for %s",
            "analyzer",
            self.metric_promql,
        )
        self.sync_new_series()

    def stop(self):
        with self.predictor_dict_lock:
            if self.is_stopped:
                return

            self.is_stopped = True

            if self.stop_event is not None:
                self.stop_event.set()
                self.logger.info(
                    "[%s] promql analyzer for %s stopped",
                    "analyzer",
                    self.metric_promql,
                )

            if self.gauge_metric is not None:
                try:
                    REGISTRY.unregister(self.gauge_metric)
                    self.logger.info(
                        "[%s] gauge metric %s unregistered",
                        "analyzer",
                        self.metric_promql,
                    )
                except KeyError:
                    self.logger.warning(
                        "[%s] gauge metric %s not registered or already unregistered",
                        "analyzer",
                        self.metric_promql,
                    )
                finally:
                    self.gauge_metric = None
