from abc import ABC, abstractmethod

class ModelPredictor(ABC):
    """docstring for MetricPredictor."""

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def predict_value(self):
        pass

    @abstractmethod
    def retrain(self):
        pass
        