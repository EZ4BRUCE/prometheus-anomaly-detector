import logging
import asyncio
import threading
import time
from engine.analyzer import MetricAnalyzer

class AnalyzeManager:
    logger: logging.Logger = None
    cluster_mode: bool = False
    analyzers: dict[str, MetricAnalyzer] = {}
    lock: threading.Lock = None
    prometheus_url: str = None
    threads: list[threading.Thread] = []

    def __init__(
        self, logger: logging.Logger, cluster_mode=False, prometheus_url: str = None
    ):
        self.logger = logger
        self.cluster_mode = cluster_mode
        self.analyzers = {}
        self.lock = threading.Lock()
        self.prometheus_url = prometheus_url

    def delete_metric(self, metric_promql: str):
        with self.lock:
            if metric_promql in self.analyzers:
                self.analyzers[metric_promql].stop()
                del self.analyzers[metric_promql]
                self.logger.info(
                    "[%s] promql analyzer for %s deleted", "manager", metric_promql
                )
            else:
                self.logger.warning(
                    "[%s] promql analyzer for %s not found", "manager", metric_promql
                )

    def add_metric(
        self,
        metric_promql: str,
        model_name: str,
        prom_url: str,
        rolling_data_window_size: str,
        retraining_interval_minutes: int,
        sync_new_series_interval_seconds: int,
    ):
        with self.lock:
            if metric_promql in self.analyzers:
                self.logger.info(
                    "[%s] promql analyzer for %s already exists, skip init",
                    "analyzer",
                    metric_promql,
                )
                return

        analyzer = MetricAnalyzer(
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
                "[%s] Adding analyzer for %s(id: %s)",
                "manager",
                metric_promql,
                id(analyzer),
            )
            self.analyzers[metric_promql] = analyzer
            self.threads.append(thread)
        time.sleep(2)
        thread.start()

    async def predict(self):

        self.logger.info(
            "[%s] predicting series values for analyzers: [%s]",
            "manager",
            ", ".join([analyzer.metric_promql for analyzer in self.analyzers.values()]),
        )

        if len(self.analyzers) == 0:
            self.logger.info("[%s] No analyzers to predict", "manager")
            return

        self.logger.info(
            "[%s] predicting series values for %s analyzers",
            "manager",
            len(self.analyzers),
        )

        tasks = []
        with self.lock:
            for analyzer in self.analyzers.values():
                tasks.append(asyncio.create_task(analyzer.predict_all_series_values()))

        await asyncio.gather(*tasks)

    def set_rolling_data_window_size(
        self, metric_promql: str, rolling_data_window_size: str
    ):
        with self.lock:
            self.analyzers[metric_promql].set_rolling_data_window_size(
                rolling_data_window_size
            )

    def set_retraining_interval_minutes(
        self, metric_promql: str, retraining_interval_minutes: int
    ):
        with self.lock:
            self.analyzers[metric_promql].set_retraining_interval_minutes(
                retraining_interval_minutes
            )

    def cleanup(self):
        """清理资源，确保所有线程在程序退出时被正确关闭。"""
        for thread in self.threads:
            if thread.is_alive():
                # 这里可以实现更复杂的逻辑来安全地停止线程
                self.logger.info("[%s] Stopping thread...", "manager")
        self.logger.info("[%s] All threads stopped.", "manager")
