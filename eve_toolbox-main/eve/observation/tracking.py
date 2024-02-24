import numpy as np

from ..intervention.intervention import Intervention
from .observation import Observation, gym


class Tracking(Observation):
    def __init__(
        self,
        intervention: Intervention,
        n_points: int = 2,
        resolution: float = 1.0,
        name: str = "tracking",
    ) -> None:
        super().__init__(name)
        self.intervention = intervention
        self.n_points = n_points
        self.resolution = resolution

    @property
    def space(self) -> gym.spaces.Box:
        low = self.intervention.tracking_space.low
        high = self.intervention.tracking_space.high
        low = np.tile(low, [self.n_points, 1])
        high = np.tile(high, [self.n_points, 1])
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self) -> None:
        self.obs = self._calculate_tracking_state()

    def reset(self, episode_nr: int = 0) -> None:
        self.step()

    def _calculate_tracking_state(self) -> np.ndarray:
        tracking = self.intervention.tracking
        inserted_length = max(self.intervention.device_lengths_inserted.values())
        return self._evenly_distributed_tracking(tracking, inserted_length)

    def _evenly_distributed_tracking(
        self, tracking: np.ndarray, inserted_length: float
    ) -> np.ndarray:

        tracking_diff = tracking[:-1] - tracking[1:]
        tracking_length = np.linalg.norm(tracking_diff, axis=-1)
        tracking_length = np.sum(tracking_length)
        if tracking_length > 0 and inserted_length > 0:
            scaled_resolution = tracking_length / inserted_length * self.resolution
        else:
            scaled_resolution = self.resolution
        tracking = list(tracking)
        tracking_state = [tracking[0]]
        if self.n_points > 1:
            acc_dist = 0.0
            for point, next_point in zip(tracking[:-1], tracking[1:]):
                if len(tracking_state) >= self.n_points or np.all(point == next_point):
                    break
                length = np.linalg.norm(next_point - point)
                dist_to_point = scaled_resolution - acc_dist
                acc_dist += length
                while (
                    acc_dist >= scaled_resolution
                    and len(tracking_state) < self.n_points
                ):
                    unit_vector = (next_point - point) / length
                    tracking_point = point + unit_vector * dist_to_point
                    tracking_state.append(tracking_point)
                    acc_dist -= scaled_resolution

            while len(tracking_state) < self.n_points:
                tracking_state.append(tracking_state[-1])
        return np.array(tracking_state, dtype=np.float32)
