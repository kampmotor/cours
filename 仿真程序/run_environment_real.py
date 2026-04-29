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
import numpy as np
import pygame
import rospy

from modules.deployment.real_env.real_cross_env import RealCrossEnvironment
from modules.deployment.utils.manager import Manager


def main():
    pygame.init()

    # env = ConfigurableEnvironment(data_file='../config/env_config.json')
    # env = CrossEnvironment(radius=450, robot_num=30, data_file='../config/env_config.json', obstacle_num=30)
    # env = FormationEnvironment(1000, 1000, robot_num=150)
    # env = PursuitEnvironment(1000, 1000, robot_num=10, obstacle_num=10)
    # env = ExploreEnvironment(1000, 1000, robot_num=4)
    # env = SheepdogEnvironment(1000, 1000, sheep_num=30, dog_num=6)
    # env = CoverEnvironment(1000, 1000, robot_num=200)
    # env = CollectEnvironment(1000, 1000, num_1=10, num_2=5, robot_num=3)
    # env = AssemblyEnvironment(1000, 1000, robot_num=10)
    # env = MoveEnvironment(1000, 1000, robot_num=6, obstacle_num=100)
    # env=MoveFormationEnvironment(1000, 1000, robot_num=5, obstacle_num=30)
    # env = RealEnvironment(data_file='../config/vicon_data.json')
    env = RealCrossEnvironment(
        data_file="../config/vicon_data.json", engine_type="QuadTreeEngine"
    )
    screen = pygame.display.set_mode(
        (env.width * env.scale_factor, env.height * env.scale_factor)
    )
    clock = pygame.time.Clock()

    manager = Manager(env)
    manager.publish_observations()

    running = True
    rate = rospy.Rate(10)
    # env.save_entities_to_file()
    draw_counter = 0
    draw_frequency = 1  # 每帧绘图一次

    # data_root = f"/home/iusl/Desktop/code_llm_ws/src/code_llm/CodeLLM/workspace/{rospy.get_param('path', 'test')}"
    # count = len(listdir(f"{data_root}/data/frames/"))  # this is used to number the 'frames' folder
    # frame_dir = f"{data_root}/data/frames/frame{count}"
    # if not os.path.exists(frame_dir):
    #     makedirs(frame_dir)

    frame_files = []

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            dt = clock.tick(10) / 1000
            env.update(dt)
            if draw_counter % draw_frequency == 0:
                env.draw(screen)
                frame = pygame.surfarray.array3d(screen).astype(np.uint8)
                frame = np.rot90(frame, 3)
                frame = np.flip(frame, axis=1)
                # frame_image_path = os.path.join(frame_dir, f'{draw_counter}.png')
                # cv2.imwrite(frame_image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                # frame_files.append(frame_image_path)

            manager.publish_observations()
            draw_counter += 1
            rate.sleep()
    finally:
        print("Shutting down")
        pygame.quit()
        env.save_entities_to_file()

    # def draw(self, screen):
    #     screen.fill((255, 255, 255))
    #     for entity in self._entities:
    #         if entity.shape == 'circle':
    #             pygame.draw.circle(screen, pygame.Color(entity.color), entity.position.astype(int), int(entity.size))
    #         else:
    #             rect = pygame.Rect(entity.position[0] - entity.size[0] / 2, entity.position[1] - entity.size[1] / 2,
    #                                entity.size[0], entity.size[1])
    #             pygame.draw.rect(screen, pygame.Color(entity.color), rect)
    #     pygame.display.flip()


if __name__ == "__main__":
    main()
