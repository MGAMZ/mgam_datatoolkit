from abc import ABC, abstractmethod
import numpy as np



class AverageStrategy(ABC):
    """平均值计算策略的抽象基类"""

    @abstractmethod
    def calculate(self, data: list[float]) -> float:
        pass


class SimpleAverage(AverageStrategy):
    def calculate(self, data: list[float]) -> float:
        return sum(data) / len(data)


class WeightedAverage(AverageStrategy):
    def __init__(self, weights: list[float] | None = None):
        self.weights = weights

    def calculate(self, data: list[float]) -> float:
        if self.weights is None:
            # 默认使用线性递增权重
            self.weights = list(range(1, len(data) + 1))
        return np.average(data, weights=self.weights[-len(data) :]).item()


class ExponentialAverage(AverageStrategy):
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha

    def calculate(self, data: list[float]) -> float:
        result = data[0]
        for value in data[1:]:
            result = self.alpha * value + (1 - self.alpha) * result
        return result


class SlidingAverage:
    def __init__(self, window_size: int, strategy: AverageStrategy = SimpleAverage()):
        if window_size <= 0:
            raise ValueError("Window size must be positive")
        self.window_size = window_size
        self.strategy = strategy
        self.data: list[float] = []

    def update(self, value: int | float) -> float:
        """添加新值并返回当前平均值"""
        if not isinstance(value, (int, float)):
            raise TypeError("Value must be numeric")

        self.data.append(float(value))
        if len(self.data) > self.window_size:
            self.data.pop(0)
        return self.calculate()

    def calculate(self) -> float:
        """计算当前窗口的平均值"""
        if not self.data:
            raise ValueError("No data available")
        return self.strategy.calculate(self.data)

    def reset(self) -> None:
        """重置数据"""
        self.data.clear()

    def get_window_data(self) -> list[float]:
        """获取当前窗口数据"""
        return self.data.copy()

    def set_strategy(self, strategy: AverageStrategy) -> None:
        """更改计算策略"""
        self.strategy = strategy
