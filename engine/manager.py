import logging
import asyncio
import threading
import time
from engine.analyzer import MetricAnalyzer

class AnalyzeManager:
    logger: logging.Logger = None
    cluster_mode: bool = False

    # (group, detection_name) -> MetricAnalyzer
    group_analyzers: dict[tuple[str, str], MetricAnalyzer] = {}
    lock: threading.Lock = None
    prometheus_url: str = None

    def __init__(
        self, logger: logging.Logger, cluster_mode=False, prometheus_url: str = None
    ):
        self.logger = logger
        self.cluster_mode = cluster_mode
        self.group_analyzers = {}
        self.lock = threading.Lock()
        self.prometheus_url = prometheus_url

    def get_all_groups(self):
        with self.lock:
            return list(set([key[0] for key in self.group_analyzers.keys()]))

    def get_all_metric_promql(self):
        with self.lock:
            return list(self.group_analyzers.keys())

    def get_detection_group(self, group_name: str):
        with self.lock:
            jobs = []
            for key in self.group_analyzers.keys():
                if key[0] == group_name:
                    analyzer = self.group_analyzers[key]
                    with analyzer.series_lock:
                        jobs.append(
                            {
                                "group": key[0],
                                "detection_name": key[1],
                                "metric": analyzer.metric_promql,
                                "model": analyzer.model_name,
                                "window_size": analyzer.rolling_data_window_size,
                                "retraining_interval_minutes": analyzer.retraining_interval_minutes,
                                "sync_new_series_interval_seconds": analyzer.sync_new_series_interval_seconds,
                            }
                        )
            return jobs

    def delete_metric_list(self, group: str, detection_names: list[str]):
        for detection_name in detection_names:
            if (group, detection_name) in self.group_analyzers:
                self.group_analyzers[(group, detection_name)].stop()
                del self.group_analyzers[(group, detection_name)]
                self.logger.info(
                    "[%s] promql analyzer for %s deleted",
                    "manager",
                    (group, detection_name),
                )
            else:
                self.logger.warning(
                    "[%s] promql analyzer for %s not found",
                    "manager",
                    (group, detection_name),
                )

    def delete_group(self, group: str):
        with self.lock:
            list_to_delete = []
            for (g,d) in self.group_analyzers.keys():
                if g == group:
                    list_to_delete.append(d)
            self.delete_metric_list(group, list_to_delete)

    def add_metric(
        self,
        group: str,
        detection_name: str,
        metric_promql: str,
        model_name: str,
        prom_url: str,
        rolling_data_window_size: str,
        retraining_interval_minutes: int,
        sync_new_series_interval_seconds: int,
    ):
        with self.lock:
            if (group, detection_name) in self.group_analyzers:
                self.logger.info(
                    "[%s] promql analyzer for %s metricpromql %s already exists, skip init",
                    "manager",
                    (group, detection_name),
                    metric_promql,
                )
                return

        analyzer = MetricAnalyzer(
            group,
            detection_name,
            self.logger,
            self.cluster_mode,
            metric_promql,
            prom_url,
            model_name,
            rolling_data_window_size,
            retraining_interval_minutes,
            sync_new_series_interval_seconds,
        )
        thread = threading.Thread(target=analyzer.run)
        thread.daemon = True
        with self.lock:
            self.logger.info(
                "[%s] Adding analyzer for group %s, detection %s promql %s(id: %s)",
                "manager",
                group,
                detection_name,
                metric_promql,
                id(analyzer),
            )
            self.group_analyzers[(group, detection_name)] = analyzer
        thread.start()

    async def predict(self):
        # 快速检查是否有分析器
        analyzer_count = len(self.group_analyzers)
        if analyzer_count == 0:
            self.logger.info("[%s] No analyzers to predict", "manager")
            return

        self.logger.info(
            "[%s] predicting series values for %s analyzers", "manager", analyzer_count
        )

        # 不加锁直接创建任务，即使列表在过程中被修改也没关系
        tasks = []
        try:
            for analyzer in self.group_analyzers.values():
                tasks.append(asyncio.create_task(analyzer.predict_all_series_values()))
        except Exception as e:
            self.logger.error(
                "[%s] Error creating prediction tasks: %s", "manager", str(e)
            )

        # 等待所有预测任务完成
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                self.logger.error("[%s] Error during prediction: %s", "manager", str(e))

    def set_rolling_data_window_size(
        self, metric_promql: str, rolling_data_window_size: str
    ):
        with self.lock:
            self.group_analyzers[metric_promql].set_rolling_data_window_size(
                rolling_data_window_size
            )

    def set_retraining_interval_minutes(
        self, metric_promql: str, retraining_interval_minutes: int
    ):
        with self.lock:
            self.group_analyzers[metric_promql].set_retraining_interval_minutes(
                retraining_interval_minutes
            )
