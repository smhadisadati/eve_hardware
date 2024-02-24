from abc import ABC
import numpy as np

from ..target.target import Target, Intervention


class InterimTarget(Target, ABC):
    def __init__(self, intervention: Intervention, threshold: float) -> None:
        super().__init__(intervention, threshold)

        self.all_coordinates: np.ndarray
