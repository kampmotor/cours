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

import os
from os import listdir, makedirs

import cv2

from modules.deployment.gymnasium_env import (
    GymnasiumBridgingEnvironment,
    GymnasiumCoveringEnvironment,
    GymnasiumExplorationEnvironment,
    GymnasiumCirclingEnvironment,
    GymnasiumCrossingEnvironment,
    GymnasiumEncirclingEnvironment,
    GymnasiumFlockingEnvironment,
    GymnasiumHerdingEnvironment,
)


def main():
    import time
    import rospy

    from modules.deployment.utils.manager import Manager

    data_root = f"../workspace/{rospy.get_param('path', 'test')}"
    count = len(
        listdir(f"{data_root}/data/frames/")
    )  # this is used to number the 'frames' folder
    frame_dir = f"{data_root}/data/frames/frame{count}"
    if not os.path.exists(frame_dir):
        makedirs(frame_dir)
    frame_files = []
    draw_counter = 0
    draw_frequency = 100

    env = GymnasiumFlockingEnvironment("../config/env_config.json")

    obs, infos = env.reset()
    manager = Manager(env)
    manager.publish_observations(infos)
    rate = rospy.Rate(env.FPS)

    start_time = time.time()  # 记录起始时间
    frame_count = 0  # 初始化帧数计数器

    while time.time() - start_time < 180:
        action = manager.robotID_velocity
        manager.clear_velocity()

        obs, reward, termination, truncation, infos = env.step(action=action)

        frame = env.render()
        if draw_counter % draw_frequency == 0:
            frame_image_path = os.path.join(frame_dir, f"{draw_counter}.png")
            cv2.imwrite(frame_image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame_files.append(frame_image_path)
        draw_counter += 1

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


if __name__ == "__main__":
    main()
