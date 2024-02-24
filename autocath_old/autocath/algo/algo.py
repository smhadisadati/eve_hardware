from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Dict
import numpy as np
from .model import Model, NetworkStatesContainer
import torch


class Algo(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self._device: torch.device = torch.device("cpu")
        self.lr_scheduler_step_counter = 0

    @property
    @abstractmethod
    def model(self) -> Model:
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...

    @abstractmethod
    def update(self, batch) -> List[float]:
        ...

    @abstractmethod
    def lr_scheduler_step(self) -> None:
        self.lr_scheduler_step_counter += 1

    @abstractmethod
    def get_exploration_action(self, flat_state: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def get_eval_action(self, flat_state: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    def copy(self):
        copy = deepcopy(self)
        return copy

    @abstractmethod
    def copy_shared_memory(self):
        ...

    @abstractmethod
    def to(self, device: torch.device):
        ...

    @abstractmethod
    def close(self):
        ...
