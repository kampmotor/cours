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

from typing import Optional

import numpy as np

from modules.deployment.entity import Obstacle, Robot
from modules.deployment.gymnasium_env.gymnasium_base_env import GymnasiumEnvironmentBase


class GymnasiumCrossingEnvironment(GymnasiumEnvironmentBase):
    def __init__(
            self, data_file: str = None, radius: float = 2.5, center: tuple = (0, 0)
    ):
        super().__init__(data_file)
        self.radius = radius
        self.center = center

    def init_entities(self):
        # obstacle_points = self.sample_points_inside_circle(self.radius, self.center, self.num_obstacles, 1.5)
        obstacle_points = (
                np.array(
                    [
                        (-1.1, -1.4),
                        (1.1, -1.4),
                        (2, 0),
                        (-2, 0),
                        (1.1, 1.4),
                        (-1.1, 1.4),
                        (0, 0),
                    ]
                )
                * 0.8
        )
        robot_points = self.sample_points_on_circle(
            self.radius, self.center, self.num_robots
        )
        robot_colors = ["green", "blue", "red", "yellow", "purple", "orange", "cyan"]
        farthest_points = self.find_farthest_points(robot_points)
        for entity_id, initial_position in enumerate(
                obstacle_points, start=len(robot_points)
        ):
            obstacle = Obstacle(
                obstacle_id=entity_id, initial_position=initial_position, size=0.15
            )
            self.add_entity(obstacle)

        for entity_id, (initial_position, target_position) in enumerate(
                zip(robot_points, farthest_points)
        ):
            robot = Robot(
                robot_id=entity_id,
                initial_position=initial_position,
                target_position=target_position,
                size=0.15,
                color=robot_colors[entity_id],
            )
            self.add_entity(robot)

    @staticmethod
    def find_farthest_points(points):
        points = np.array(points)
        distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)
        farthest_indices = np.argmax(distances, axis=1)
        return points[farthest_indices]

    @staticmethod
    def sample_points_on_circle(radius, center, num_points):
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        points = [
            (center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))
            for angle in angles
        ]
        return points

    @staticmethod
    def sample_points_inside_circle(
            radius, center, num_points, min_distance, max_attempts_per_point=10
    ):
        def distance(p1, p2):
            return np.sqrt(np.sum((p1 - p2) ** 2, axis=1))

        points = []
        center = np.array(center)
        attempts = 0

        radius -= min_distance
        while (
                len(points) < num_points and attempts < num_points * max_attempts_per_point
        ):
            # Generate random points within the bounding square
            random_points = (
                    np.random.uniform(-radius, radius, size=(num_points, 2)) + center
            )
            valid_mask = np.linalg.norm(random_points - center, axis=1) <= radius
            random_points = random_points[valid_mask]

            for point in random_points:
                if len(points) == 0 or np.all(
                        distance(np.array(points), point) >= min_distance
                ):
                    points.append(point)
                    if len(points) >= num_points:
                        break

            attempts += 1

        if len(points) < num_points:
            print(
                f"Warning: Could only place {len(points)} points out of {num_points} requested."
            )

        return np.array(points)


if __name__ == "__main__":
    env = GymnasiumCrossingEnvironment("../../../config/env/crossing_config.json")
    obs, infos = env.reset()

    from modules.deployment.utils.manager import Manager

    manager = Manager(env, max_speed=1.0)
    manager.publish_observations(infos)
    import rospy

    rate = rospy.Rate(env.FPS)
    while True:
        action = manager.robotID_velocity
        manager.clear_velocity()
        obs, reward, termination, truncation, infos = env.step(action=action)
        env.render()
        manager.publish_observations(infos)
        rate.sleep()
    print("Simulation completed successfully.")
