from abc import ABC, abstractmethod

class SeriesPredictor(ABC):
    """docstring for MetricPredictor."""

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass
        
    @abstractmethod
    def get_model_name(self):
        pass

    @abstractmethod
    def get_model_description(self):
        pass

    @abstractmethod
    def get_series_hash(self):
        pass
