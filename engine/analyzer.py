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
    
    group:str = None
    detection_name:str = None

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
    series_lock: threading.Lock = None

    background_thread: threading.Thread = None
    stop_event: threading.Event = None

    is_stopped: bool = False

    def __init__(
        self,
        group,
        detection_name,
        logger,
        cluster_mode,
        metric_promql,
        prometheus_url,
        model_name,
        rolling_data_window_size,
        retraining_interval_minutes,
        sync_new_series_interval_seconds,
    ):
        self.group = group
        self.detection_name = detection_name
        self.series_predictors = {}
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
        self.series_lock = threading.Lock()
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
        
        # 快速检查是否有预测器
        if len(self.series_predictors) == 0:
            self.logger.warning(
                "[%s] %s(id: %s) no series to predict",
                "analyzer",
                self.metric_promql,
                id(self),
            )
            return

        # 不加锁直接预测，即使数据可能稍旧也没关系
        should_delete_predictors = []
        for hash, predictor in self.series_predictors.items():
            try:
                ok = predictor.predict(now)
                if not ok:
                    should_delete_predictors.append(hash)
            except Exception as e:
                self.logger.error(
                    "[%s] Error predicting for series %s: %s",
                    "analyzer",
                    hash,
                    str(e)
                )

        # 只在需要删除时加锁
        if should_delete_predictors:
            with self.series_lock:
                for hash in should_delete_predictors:
                    if hash in self.series_predictors:
                        predictor = self.series_predictors[hash]
                        try:
                            # 清理相关的指标
                            for value_type in ["yhat", "anomaly", "yhat_upper", "yhat_lower"]:
                                delete_series = {
                                    **predictor.metric.label_config,
                                    "value_type": value_type,
                                    "model_name": predictor.model_name,
                                    "metric_type": "anomaly-detection",
                                    "origin_metric_name": predictor.metric.metric_name,
                                }
                                self.gauge_metric.remove(
                                    *self.gauge_metric.labels(**delete_series)._labelvalues
                                )
                            del self.series_predictors[hash]
                        except Exception as e:
                            self.logger.error(
                                "[%s] Error cleaning up predictor %s: %s",
                                "analyzer",
                                hash,
                                str(e)
                            )

    async def check_and_retrain_predictors(self):
        """Asynchronously retrain the predictors that need updating."""
        current_time = datetime.now()

        # 过滤出需要重新训练的预测器
        with self.series_lock:
            retrain_predictors = [
                predictor
                for predictor in self.series_predictors.values()
                if (current_time - predictor.last_retrain_time).total_seconds() / 60
                >= self.retraining_interval_minutes
            ]

        if not retrain_predictors:
            self.logger.info(
                "[%s] Metric %s has no predictors need retraining at this time.",
                "analyzer",
                self.metric_promql,
            )
            return

        self.logger.info(
            "[%s] Retraining metric %s's %s predictors.",
            "analyzer",
            self.metric_promql,
            len(retrain_predictors),
        )

        # 直接将 retrain_predictors 列表传递给 train_model_async
        await self.train_model_async(retrain_predictors, initial_run=False)

        # 更新 last_retrain_time
        for predictor in retrain_predictors:
            predictor.last_retrain_time = current_time

    def series_data_ready(self, metric_name, labels, time_range) -> bool:
        data_start_time = datetime.now() - parse_timedelta("now", time_range)
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
                self.logger.info("metric: %s no series to add", self.metric_promql)
                return

            self.logger.info(
                "[%s] Metric %s got %s series total",
                "analyzer",
                self.metric_promql,
                len(current_series),
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
                        "[%s] metric %s series %s has no %s",
                        "analyzer",
                        self.metric_promql,
                        series["metric"],
                        CONST_METRIC_NAME_LABEL_KEY,
                    )

                metric_name = series["metric"][CONST_METRIC_NAME_LABEL_KEY]
                self.metric_name = metric_name

                labels = deepcopy(series["metric"])
                del labels[CONST_METRIC_NAME_LABEL_KEY]

                if not self.same_label_keys(labels.keys()):
                    self.logger.warning(
                        "[%s] metric %s series %s label keys not match: %s",
                        "analyzer",
                        self.metric_promql,
                        series["metric"],
                        labels.keys(),
                    )
                    continue

                if not self.series_data_ready(
                    metric_name, labels, self.rolling_data_window_size
                ):
                    self.logger.warning(
                        "[%s] metric: %s series: %s data is not ready(time window: %s) to train, skip training",
                        "analyzer",
                        self.metric_promql,
                        series["metric"],
                        self.rolling_data_window_size,
                    )
                    continue
                else:
                    self.logger.info(
                        "[%s] metric: %s series: %s data is ready(time window: %s) to train, start training",
                        "analyzer",
                        self.metric_promql,
                        series["metric"],
                        self.rolling_data_window_size,
                    )

                series_label_hash = hash(frozenset(labels.items()))
                with self.series_lock:
                    if series_label_hash not in self.series_predictors:
                        self.logger.info(
                            "[%s] metric: %s series: %s got new series: %s",
                            "analyzer",
                            self.metric_promql,
                            series["metric"],
                            series,
                        )
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

    async def sync_new_series_async(self):
        try:
            # Get current metric value
            current_series = self.prometheus_client.get_current_metric_value(
                metric_name=self.metric_promql
            )

            if len(current_series) == 0:
                self.logger.info("metric: %s no series to add", self.metric_promql)
                return

            self.logger.info(
                "[%s] metric: %s got %s series total",
                "analyzer",
                self.metric_promql,
                len(current_series),
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
                        "[%s] metric: %s series: %s label keys not match: %s",
                        "analyzer",
                        self.metric_promql,
                        series["metric"],
                        labels.keys(),
                    )
                    continue

                data_is_ready = self.series_data_ready(
                    metric_name, labels, self.rolling_data_window_size
                )

                series_label_hash = hash(frozenset(labels.items()))
                with self.series_lock:
                    if series_label_hash not in self.series_predictors:
                        self.logger.info("[%s] got new series: %s", "analyzer", series)
                        if not data_is_ready:
                            self.logger.warning(
                                "[%s] metric: %s series: %s data is not ready (time window: %s) to train, skip training",
                                "analyzer",
                                self.metric_promql,
                                series["metric"],
                                self.rolling_data_window_size,
                            )
                            continue
                        else:
                            self.logger.info(
                                "[%s] metric: %s series: %s data is ready (time window: %s) to train, start training",
                                "analyzer",
                                self.metric_promql,
                                series["metric"],
                                self.rolling_data_window_size,
                            )
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
                await self.train_model_async(new_predictors, initial_run=True)

            else:
                self.logger.info(
                    "[%s] Metric [%s] all series' predictors already exists. Skipping training",
                    "analyzer",
                    self.metric_name,
                )
        except Exception as e:
            self.logger.error(
                "[%s] Error syncing new series for metric: %s: %s",
                "analyzer",
                self.metric_promql,
                str(e),
            )

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
            self.logger.warning(
                "[%s] Metric %s has no series to train. Skipping training.",
                "analyzer",
                self.metric_promql,
            )
            return

        self.logger.info(
            "[%s] Training models asynchronously for metric %s's %d new series",
            "analyzer",
            self.metric_promql,
            len(predictors),
        )

        # Create asynchronous tasks for each predictor
        tasks = [
            self.train_individual_model_async(predictor, initial_run)
            for predictor in predictors
        ]

        with self.series_lock:
            if self.is_stopped:
                self.logger.info(
                    "[%s] promql analyzer for metric %s already stopped",
                    "analyzer",
                    self.metric_promql,
                )
                return

        # Run all tasks concurrently
        result = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and failed results
        valid_predictors = [
            predictor for predictor in result 
            if predictor is not None and not isinstance(predictor, Exception)
        ]

        # Log any exceptions that occurred
        for r in result:
            if isinstance(r, Exception):
                self.logger.error(
                    "[%s] Error during model training: %s",
                    "analyzer",
                    str(r)
                )

        # Update global PREDICTOR_MODEL_LIST
        with self.series_lock:
            if len(valid_predictors) == 0:
                self.logger.info(
                    "[%s] Metric %s has no predictor trained successfully",
                    "analyzer",
                    self.metric_promql,
                )
                return

            for predictor in valid_predictors:
                if predictor is not None:
                    self.series_predictors[predictor.get_series_hash()] = predictor

            self.logger.info(
                "[%s] Metric %s has %d predictors added",
                "analyzer",
                self.metric_promql,
                len(self.series_predictors),
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
            self.logger.error(
                "[%s] Error training model for metric %s series %s: %s",
                "analyzer",
                self.metric_promql,
                predictor_model.metric.label_config,
                str(e),
            )
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
            "[%s] Total Training time taken = %s, for metric: %s series: %s",
            "analyzer",
            str(datetime.now() - start_time),
            self.metric_promql,
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
        else:
            self.logger.info(
                "This analyzer is already running, skip initial training"
            )
            return

        # 为每个schedule添加唯一标识
        retrain_job_tag = f"retrain_predictors_{self.group}_{self.detection_name}"
        sync_job_tag = f"sync_series_{self.group}_{self.detection_name}"
        
        # 检查是否已存在相同标记的任务
        existing_jobs = [job for job in schedule.get_jobs() if job.tags]
        existing_tags = [tag for job in existing_jobs for tag in job.tags]

        # 只在任务不存在时创建新的schedule
        if retrain_job_tag not in existing_tags:
            schedule.every(90).seconds.do(
                lambda: asyncio.run(self.check_and_retrain_predictors())
            ).tag(retrain_job_tag)

            self.logger.info(
                "[%s] Scheduled check predictors retrain schedule every 90 seconds.",
                "analyzer",
            )

        if sync_job_tag not in existing_tags:
            schedule.every(self.sync_new_series_interval_seconds).seconds.do(
                lambda: asyncio.run(self.resync_series())
            ).tag(sync_job_tag)

            self.logger.info(
                "[%s] Scheduled sync series every %s seconds for metric: %s",
                "analyzer",
                self.sync_new_series_interval_seconds,
                self.metric_promql,
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
        await self.sync_new_series_async()

    def stop(self):
        with self.series_lock:
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
