"""Abstract class for model"""
from abc import ABC, abstractmethod


class SuperResolutionModelBase(ABC):
    """Abstract class for models"""

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass
