"""Abstract class for model"""
from abc import ABC, abstractmethod


class SuperResolutionModel(ABC):
    """Abstract class for models"""

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self):
        pass