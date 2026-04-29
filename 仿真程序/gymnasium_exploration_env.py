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

from modules.deployment.entity import Landmark, Robot, Obstacle
from modules.deployment.utils.sample_point import *
from modules.deployment.gymnasium_env.gymnasium_base_env import GymnasiumEnvironmentBase
from modules.deployment.utils.save import save_frames_as_animations

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")


class GymnasiumExplorationEnvironment(GymnasiumEnvironmentBase):
    def __init__(self, data_file: str):
        super().__init__(data_file)

    def init_entities(self):
        entity_id = 0

        for x in np.arange(-self.width * 0.4, self.width * 0.51, 0.2 * self.width):
            for y in np.arange(
                -self.height * 0.4, self.height * 0.51, 0.2 * self.height
            ):
                landmark = Landmark(
                    landmark_id=entity_id,
                    initial_position=(x, y),
                    size=np.array([0.2 * self.width, 0.2 * self.height]),
                    color="gray",
                )
                self.add_entity(landmark)
                entity_id += 1
        # landmark = Landmark(landmark_id=entity_id,
        #                     initial_position=(0,0),
        #                     size=np.array([0.1 * self.width, 0.1 * self.height]),
        #                     color='gray')
        # self.add_entity(landmark)
        # entity_id += 1

        robot_size = self.data["entities"]["robot"]["size"]
        shape = self.data["entities"]["robot"]["shape"]
        color = self.data["entities"]["robot"]["color"]
        self.num_robots = 3
        for i in range(self.num_robots):
            position = np.array((-0.25, -0.25)) + 0.25 * i
            robot = Robot(
                robot_id=entity_id,
                initial_position=position,
                target_position=None,
                size=robot_size,
                color=color,
            )
            entity_id += 1
            self.add_entity(robot)

    def step(self, action: ActType):
        obs, reward, termination, truncation, infos = super().step(action)

        for entity in self.entities:
            if isinstance(entity, Robot):
                for landmark in self.entities:
                    if isinstance(landmark, Landmark):
                        if self.is_robot_within_landmark(entity, landmark):
                            landmark.color = "blue"
                            landmark.state = "visited"

        return obs, reward, termination, truncation, infos

    def init_omni_entities(self):
        robot_id_list = self.robot_id_list
        target_positions = [
            (2.0, 0.0),
            (1.7320508075688774, 0.9999999999999999),
            (1.0000000000000002, 1.7320508075688772),
            (1.2246467991473532e-16, 2.0),
            (-0.9999999999999996, 1.7320508075688774),
            (-1.732050807568877, 1.0000000000000007),
        ]

        for count, i in enumerate(robot_id_list):
            robot = Robot(
                robot_id=i,
                initial_position=(0, 0),
                size=0.15,
                target_position=target_positions[count],
            )
            self.add_entity(robot)
        obstacle_id_list = self.obstacle_id_list
        for i in obstacle_id_list:
            obstacle = Obstacle(
                obstacle_id=i, initial_position=(0, 0), size=0.15, movable=True
            )
            self.add_entity(obstacle)
        landmark_id_list = self.landmark_id_list
        for i in landmark_id_list:
            landmark = Landmark(
                landmark_id=i, initial_position=(0, 0), size=0.15, color="gray"
            )
            self.add_entity(landmark)
        entity_id = 10
        for x in np.arange(-self.width * 0.4, self.width * 0.51, 0.2 * self.width):
            for y in np.arange(
                -self.height * 0.4, self.height * 0.51, 0.2 * self.height
            ):
                landmark = Landmark(
                    landmark_id=entity_id,
                    initial_position=(x, y),
                    size=np.array([0.2 * self.width, 0.2 * self.height]),
                    color="gray",
                )
                self.add_entity(landmark)
                entity_id += 1
        prey_id_list = self.prey_id_list
        for i in prey_id_list:
            prey = Prey(
                prey_id=i,
                initial_position=(0, 0),
                size=0.15,
            )
            self.add_entity(prey)

    def is_robot_within_landmark(self, robot: Robot, landmark: Landmark):
        if np.all(
            np.linalg.norm(robot.position - landmark.position)
            < (robot.size + 0.5 * landmark.size)
        ):
            return True
        return False


if __name__ == "__main__":
    import time
    import rospy

    from modules.deployment.utils.manager import Manager

    env = GymnasiumExplorationEnvironment("../../../config/env/exploration_config.json")

    obs, infos = env.reset()
    manager = Manager(env)
    manager.publish_observations(infos)
    rate = rospy.Rate(env.FPS)
    frames = []
    start_time = time.time()  # 记录起始时间
    frame_count = 0  # 初始化帧数计数器

    while not rospy.is_shutdown():
        action = manager.robotID_velocity
        # action = {}
        # manager.clear_velocity()
        obs, reward, termination, truncation, infos = env.step(action=action)
        env.render()
        # frames.append(env.render())
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
    # save_frames_as_animations(0, '../../../workspace/exploration/pic', frames)
