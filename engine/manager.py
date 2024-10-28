import logging
import asyncio
import threading

from engine.analyzer import MetricAnalyzer

class AnalyzeManager:
    logger: logging.Logger = None
    cluster_mode: bool = False
    analyzers: dict[str, MetricAnalyzer] = {}
    lock: threading.Lock = None

    def __init__(self, logger: logging.Logger, cluster_mode=False):
        self.logger = logger
        self.cluster_mode = cluster_mode
        self.analyzers = {}
        self.lock = threading.Lock()
        
    def delete_metric(self, metric_promql: str):
        with self.lock:
            if metric_promql in self.analyzers:
                self.analyzers[metric_promql].stop()
                del self.analyzers[metric_promql]
                self.logger.info("promql analyzer for %s deleted", metric_promql)
            else:
                self.logger.warning("promql analyzer for %s not found", metric_promql)

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
                    "promql analyzer for %s already exists, skip init", metric_promql
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

        with self.lock:
            self.analyzers[metric_promql] = analyzer
        
        thread = threading.Thread(target=analyzer.run)
        thread.daemon = True 
        thread.start()
        
    async def predict(self):
        if len(self.analyzers) == 0:
            self.logger.info("No analyzers to predict")
            return
        
        self.logger.info("predicting series values for %s analyzers", len(self.analyzers))
        
        tasks = []
        with self.lock:
            for analyzer in self.analyzers.values():
                tasks.append(asyncio.create_task(analyzer.predict_all_series_values()))
        
        await asyncio.gather(*tasks)

    def set_rolling_data_window_size(self, metric_promql: str, rolling_data_window_size: str):
        with self.lock:
            self.analyzers[metric_promql].set_rolling_data_window_size(rolling_data_window_size)

    def set_retraining_interval_minutes(self, metric_promql: str, retraining_interval_minutes: int):
        with self.lock:
            self.analyzers[metric_promql].set_retraining_interval_minutes(retraining_interval_minutes)
