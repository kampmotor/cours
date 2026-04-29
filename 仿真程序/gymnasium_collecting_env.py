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

from typing import Optional, TypeVar

from modules.deployment.entity import Landmark, PushableObject, Robot
from modules.deployment.utils.sample_point import *
from modules.deployment.gymnasium_env.gymnasium_base_env import GymnasiumEnvironmentBase

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")


class GymnasiumCollectingEnvironment(GymnasiumEnvironmentBase):
    def __init__(self, data_file: str, num_1, num_2):
        super().__init__(data_file)
        self.entity_1_num = num_1
        self.entity_2_num = num_2

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        self.init_entities()
        obs = self.get_observation("array")
        infos = self.get_observation("dict")
        return obs, infos

    def init_entities(self):
        entity_id = 0

        range_a = Landmark(
            landmark_id=entity_id,
            initial_position=(-0.5, -1),
            size=np.array((1, 2)),
            color="gray",
        )

        range_b = Landmark(
            landmark_id=entity_id + 1,
            initial_position=(1.5, 1),
            size=np.array((2, 1)),
            color="red",
        )

        self.add_entity(range_a)
        self.add_entity(range_b)

        entity_id = 2

        robot_size = self.data["entities"]["robot"]["size"]
        robot_num = self.data["entities"]["robot"]["count"] - len(
            self.data["entities"]["robot"]["specified"]
        )
        robot_shape = self.data["entities"]["robot"]["shape"]
        robot_color = self.data["entities"]["robot"]["color"]

        for i in range(self.num_robots):
            position = sample_point(
                zone_center=[0, 0],
                zone_shape="rectangle",
                zone_size=[self.width, self.height],
                robot_size=robot_size,
                robot_shape=robot_shape,
                min_distance=robot_size,
                entities=self.entities,
            )
            robot = Robot(
                robot_id=entity_id,
                initial_position=position,
                size=robot_size,
                color=robot_color,
            )
            self.add_entity(robot)
            entity_id += 1

        for i in range(self.entity_1_num):
            position = sample_point(
                zone_center=[0, 0],
                zone_shape="rectangle",
                zone_size=[self.width, self.height],
                robot_size=robot_size,
                robot_shape=robot_shape,
                min_distance=robot_size,
                entities=self.entities,
            )
            object = PushableObject(
                object_id=entity_id, initial_position=position, size=0.1, color="red"
            )
            object.density = 0.01

            self.add_entity(object)
            entity_id += 1

        for i in range(self.entity_2_num):
            position = sample_point(
                zone_center=[0, 0],
                zone_shape="rectangle",
                zone_size=[self.width, self.height],
                robot_size=robot_size,
                robot_shape=robot_shape,
                min_distance=robot_size,
                entities=self.entities,
            )
            object = PushableObject(
                object_id=entity_id, initial_position=position, size=0.1, color="yellow"
            )
            object.density = 0.01
            self.add_entity(object)
            entity_id += 1

    def step(self, action: ActType):
        obs, reward, termination, truncation, infos = super().step(action)
        for entity in self.entities:
            if isinstance(entity, PushableObject):
                if entity.position[0] < -100 or entity.position[0] > 900:
                    self.remove_entity(entity.id)
            # if entity.moveable:
            #     entity.position = state[0][entity.id]
            #     entity.velocity = state[1][entity.id]
        return obs, reward, termination, truncation, infos


if __name__ == "__main__":
    import time
    import rospy

    from modules.deployment.utils.manager import Manager

    env = GymnasiumCollectingEnvironment("../../../config/env_config.json", 10, 10)

    obs, infos = env.reset()
    manager = Manager(env)
    manager.publish_observations(infos)
    rate = rospy.Rate(env.FPS)

    start_time = time.time()  # 记录起始时间
    frame_count = 0  # 初始化帧数计数器

    while True:
        # action = manager.robotID_velocity
        action = {}
        # manager.clear_velocity()
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
