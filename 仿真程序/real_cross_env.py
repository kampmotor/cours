"""
Copyright (c) 2024 WindyLab of Westlake University, China
All rights reserved.

This software is provided "as is" without warranty of any kind, either
express or implied, including but not limited to the warranties of
merchantability, fitness for a particular purpose, or non-infringement.
In no event shall the authors or copyright holders be liable for any
claim, damages, or other liability, whether in an action of contract,
tort, or otherwise, arising from, out of, or in connection with the
software or the use or other dealings in the software.
"""

import json

import numpy as np
import pygame

from modules.deployment.entity import Leader, Obstacle, Robot
from .gymnasium_real_env import GymnasiumRealEnvironment


class RealCrossEnvironment(GymnasiumRealEnvironment):
    def __init__(
        self,
        data_file: str = None,
        engine_type="OmniEngine",
        output_file: str = "output.json",
    ):
        super().__init__(data_file=data_file)
        self.output_file = output_file
        self.init_entities()

    def init_entities(self):
        robots_pos = [entity.position for entity in self.get_entities_by_type("Robot")]
        target_pos = self.find_farthest_points(robots_pos)
        for index, robot in enumerate(self.get_entities_by_type("Robot")):
            robot.target_position = target_pos[index]

    @staticmethod
    def find_farthest_points(points):
        points = np.array(points)
        distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)
        farthest_indices = np.argmax(distances, axis=1)
        return points[farthest_indices]
