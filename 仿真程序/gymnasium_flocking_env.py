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

from typing import Any, SupportsFloat, TypeVar, Optional

from modules.deployment.engine import QuadTreeEngine
from modules.deployment.entity import Landmark, Leader, Obstacle, PushableObject, Robot
from modules.deployment.utils.sample_point import *
from modules.deployment.gymnasium_env.gymnasium_base_env import GymnasiumEnvironmentBase

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class GymnasiumFlockingEnvironment(GymnasiumEnvironmentBase):
    def __init__(self, data_file: str = None):
        super().__init__(data_file)
        if self.engine.__class__.__name__ == "QuadTreeEngine":
            self.engine = QuadTreeEngine(
                world_size=(self.width, self.height),
                alpha=0.5,
                damping=0.75,
                collision_check=True,
                joint_constraint=False,
            )

    def init_entities(self):
        def add_specified_entities(entity_type, entity_class, color=None):
            nonlocal entity_id
            for entity_data in self.data["entities"][entity_type]["specified"]:
                entity = entity_class(
                    entity_id, entity_data["position"], entity_data["size"]
                )
                if color:
                    entity.color = entity_data.get("color", color)
                self.add_entity(entity)
                entity_id += 1

        entity_id = 0

        add_specified_entities("leader", Leader, "red")
        add_specified_entities("obstacle", Obstacle)
        add_specified_entities("landmark", Landmark)
        add_specified_entities("pushable_object", PushableObject)
        add_specified_entities("robot", Robot, "green")

        obstacle_list = [(0, 1.6), (0, -1.8), (1.8, 0), (-1.6, 0)]
        for pos in obstacle_list:
            obstacle = Obstacle(entity_id, pos, 0.15)
            self.add_entity(obstacle)
            entity_id += 1

        # Add remaining robots
        if "count" in self.data["entities"]["robot"]:
            robot_size = self.data["entities"]["robot"]["size"]
            robot_num = self.data["entities"]["robot"]["count"] - len(
                self.data["entities"]["robot"]["specified"]
            )
            shape = self.data["entities"]["robot"]["shape"]
            color = self.data["entities"]["robot"]["color"]

            for i in range(robot_num):
                position = sample_point(
                    zone_center=[0, 0],
                    zone_shape="rectangle",
                    zone_size=[0.3 * self.width, 0.3 * self.height],
                    robot_size=robot_size,
                    robot_shape=shape,
                    min_distance=0.1,
                    entities=self.entities,
                )
                robot = Robot(entity_id, position, robot_size, color=color)
                self.add_entity(robot)
                entity_id += 1

        # if "count" in self.data["entities"]["obstacle"]:
        #     size = self.data["entities"]["obstacle"]["size"]
        #     num = self.data["entities"]["obstacle"]["count"]
        #
        #     for i in range(num):
        #         position = sample_point(zone_center=[0, 0], zone_shape='rectangle',
        #                                 zone_size=[0.8 * self.width, 0.8 * self.height],
        #                                 robot_size=size, robot_shape='circle', min_distance=size,
        #                                 entities=self.entities)
        #         robot = Obstacle(entity_id, position, size)
        #         self.add_entity(robot)
        #         entity_id += 1

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return super().step(action)


if __name__ == "__main__":
    import time
    import rospy

    from modules.deployment.utils.manager import Manager

    # env = GymnasiumFlockingEnvironment("../../../config/real_env/flocking_config.json")
    env = GymnasiumFlockingEnvironment("../../../config/env/flocking_config.json")
    obs, infos = env.reset()
    print(env.movable_agents)
    manager = Manager(env)
    manager.publish_observations(infos)
    rate = rospy.Rate(env.FPS)

    start_time = time.time()  # 记录起始时间
    frame_count = 0  # 初始化帧数计数器
    current_time = time.time()
    while current_time - start_time < 120:
        action = manager.robotID_velocity
        # manager.clear_velocity()
        # print(action)
        obs, reward, termination, truncation, infos = env.step(action=action)
        env.render()
        manager.publish_observations(infos)
        rate.sleep()

        frame_count += 1  # 增加帧数计数器
        current_time = time.time()  # 获取当前时间
        elapsed_time = current_time - start_time  # 计算已过去的时间

        # 当达到1秒时，计算并打印FPS，然后重置计数器和时间
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")  # 打印FPS，保留两位小数
            frame_count = 0  # 重置帧数计数器
            start_time = current_time  # 重置起始时间戳
    print("Simulation completed successfully.")
