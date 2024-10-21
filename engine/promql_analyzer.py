import time
import requests
import logging
from prometheus_api_client import PrometheusConnect
from engine.model_predictor.model_predictor import ModelPredictor

class PromQLAnalyzer:
    logger: logging.Logger = None
    promql: str = None
    metric_predictors: list[ModelPredictor] = None
    prometheus_client: PrometheusConnect = None
    
    def __init__(self, promql, prometheus_url):
        self.promql = promql
        self.metric_predictors = []

    def fetch_series(self):
        # 从 Prometheus 获取 series
        response = requests.get(f"{self.prometheus_url}/api/v1/query", params={'query': self.promql})
        if response.status_code == 200:
            data = response.json()
            return [result['metric'] for result in data['data']['result']]
        else:
            print("Failed to fetch data from Prometheus")
            return []

    def sync_series(self):
        pass

    def train_model(self):
        pass
    
    def run(self):
        pass
    
    def get_predicted_value(self):
        perdiction=None
        return perdiction
    

