from typing import Optional, List
import random
import numpy as np

from .target import Target, gym
from ..vesseltree import VesselTree
from ..intervention.intervention import Intervention


class CenterlineRandom(Target):
    def __init__(
        self,
        vessel_tree: VesselTree,
        intervention: Intervention,
        threshold: float,
        branches: Optional[List[str]] = None,
        min_distance_between_possible_targets: Optional[float] = None,
    ) -> None:
        super().__init__(intervention, threshold)
        self.vessel_tree = vessel_tree
        self.branches = branches
        self.min_distance_between_possible_targets = (
            min_distance_between_possible_targets
        )

        self._potential_targets = None
        self._initialized_vesseltree = None

    @property
    def coordinate_space(self) -> gym.spaces.Box:
        low = self.vessel_tree.coordinate_space.low
        high = self.vessel_tree.coordinate_space.high
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, episode_nr=0) -> None:
        if self._initialized_vesseltree != self.vessel_tree.branches:
            self._init_centerline_point_cloud()
            self._initialized_vesseltree = self.vessel_tree.branches
        self.coordinates = random.choice(self._potential_targets)
        self.reached = False

    def _init_centerline_point_cloud(self):
        self._potential_targets = None
        if self.branches is None:
            branch_keys = self.vessel_tree.keys()
        else:
            branch_keys = set(self.branches) & set(self.vessel_tree.keys())
        self._potential_targets = np.empty((0, 3))
        for branch in branch_keys:
            points = self.vessel_tree[branch].coordinates
            self._potential_targets = np.vstack((self._potential_targets, points))
