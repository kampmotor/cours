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

from .base_engine import Engine
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
import numpy as np
import socket
import os
import time
import threading

from scipy.spatial.transform import Rotation as R

from modules.deployment.utils.mqtt_pub import MqttClientThread
from sensor_msgs.msg import Joy  # 新增导入 Joy 消息类型
from modules.utils import rich_print


class OmniEngine(Engine):
    def __init__(self):
        super().__init__()
        rospy.init_node("omni_engine", anonymous=True)

        self.type_mapping = {"robot": "VSWARM", "obstacle": "OBSTACLE", "prey": "PREY"}
        self.subscribers = []
        self.mqtt_client = self.start_up_mqtt_thread()
        self.led_init = False
        self.joy_input = {"x": 0.0, "y": 0.0, "theta": 0.0}
        self.joy_timeout = 0.1
        self.last_joy_input_time = rospy.Time.now()
        self.joy_subscriber = rospy.Subscriber("/joy", Joy, self.joy_callback)

    def start_up_mqtt_thread(self):
        broker_ip = "10.0.2.66"
        port = 1883
        keepalive = 60  # 与代理通信之间允许的最长时间段（以秒为单位）
        client_id = f"{self.__class__.__name__}"  # 客户端id不能重复

        try:
            broker = os.environ["REMOTE_SERVER"]
        except KeyError:
            broker = broker_ip

        net_status = -1

        while net_status != 0:
            net_status = os.system(f"ping -c 4 {broker}")
            time.sleep(2)

        # 启动MQTT客户端线程
        mqtt_client_instance = MqttClientThread(
            broker=broker, port=port, keepalive=keepalive, client_id=client_id
        )
        mqtt_thread = threading.Thread(target=mqtt_client_instance.run)
        mqtt_thread.start()
        return mqtt_client_instance

    def pose_callback(self, msg, args):
        entity_id, entity_type = args
        position = np.array([msg.pose.position.x, msg.pose.position.y])

        self.set_position(entity_id, position)

        # print(f"update position of {entity_type} {entity_id} to {position}")
        quaternion = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        )

        rot_mat = R.from_quat(quaternion).as_matrix()
        # 解出欧拉角（RPY顺序）
        euler = np.array(R.from_matrix(rot_mat).as_euler("xyz", degrees=False))
        self.set_yaw(entity_id, euler[2])

    def twist_callback(self, msg, args):
        entity_id, entity_type = args
        velocity = np.array([msg.twist.linear.x, msg.twist.linear.y])
        entity_id = int(entity_id)
        self.set_velocity(entity_id, velocity)

    def joy_callback(self, joy_msg):
        self.joy_input["x"] = joy_msg.axes[1] * -0.25  # 前后平移（线速度X轴）
        self.joy_input["y"] = joy_msg.axes[0] * -0.25  # 左右平移（线速度Y轴）
        self.joy_input["theta"] = joy_msg.axes[3] * 2  # 左右旋转（角速度Z轴）
        self.last_joy_input_time = rospy.Time.now()  # 更新最近的输入时间
        print(f"Joy input updated: {self.joy_input}")

    def generate_subscribers(self, entity_id, entity_type):
        pose_topic = f"/vrpn_client_node/{entity_type.upper()}{entity_id}/pose"
        twist_topic = f"/vrpn_client_node/{entity_type.upper()}{entity_id}/twist"

        self.subscribers.append(
            rospy.Subscriber(
                pose_topic,
                PoseStamped,
                self.pose_callback,
                callback_args=(entity_id, entity_type),
            )
        )
        self.subscribers.append(
            rospy.Subscriber(
                twist_topic,
                TwistStamped,
                self.twist_callback,
                callback_args=(entity_id, entity_type),
            )
        )

    def generate_all_subscribers(self):
        for entity_id, entity in self._entities.items():
            a = entity.__class__.__name__.lower()
            self.generate_subscribers(
                entity_id=entity_id, entity_type=self.type_mapping[a]
            )

    def step(self, delta_time: float):
        if len(self.subscribers) == 0:
            self.generate_all_subscribers()

        # 应用遥控器的输入数据x`
        self.apply_joy_control()
        # for entity in self._entities:
        #     self.control_yaw(entity, desired_yaw=0)
        # if not self.led_init:
        self.update_led_color()
        # 继续执行原有的周期行为
        rospy.sleep(delta_time)

    def apply_force(self, entity_id: int, force: np.ndarray):
        print(f"Failed Applying force {force} to entity {entity_id} at omni bot")

    def control_velocity(self, entity_id, desired_velocity, dt=None):
        # 使用遥控器的输入控制机器人
        json_msg = {
            "x": desired_velocity["x"],
            "y": desired_velocity["y"],
            "theta": desired_velocity["theta"],
        }
        json_str = json.dumps(json_msg)
        self.mqtt_client.publish(
            f"/VSWARM{entity_id}_robot/motion", json_str.encode("utf-8")
        )
        print(f"Controlling velocity of entity {entity_id} to {json_msg} via MQTT")

    def apply_joy_control(self):
        # current_time = rospy.Time.now()
        # if (current_time - self.last_joy_input_time).to_sec() > self.joy_timeout:
        #     self.joy_input = {"x": 0.0, "y": 0.0, "theta": 0.0}
        #     print(f"Joy input timeout, resetting input to: {self.joy_input}")

        for entity in self._entities.values():
            if entity.__class__.__name__.lower() == "prey":
                self.control_velocity(entity.id, self.joy_input)

    def control_yaw(self, entity_id, desired_yaw, dt=None):
        yaw_error = desired_yaw - self._entities[entity_id].yaw
        if yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        if yaw_error < -np.pi:
            yaw_error += 2 * np.pi
        if abs(yaw_error) < 0.1:
            return
        kp = 0.8
        json_msg = {"x": 0, "y": 0, "theta": yaw_error * kp}
        print(f"yaw error is {yaw_error}")
        json_str = json.dumps(json_msg)
        self.mqtt_client.publish(
            f"/VSWARM{entity_id}_robot/motion", json_str.encode("utf-8")
        )

    def update_led_color(self):
        self.led_init = True
        color_mapping = {
            "red": 0xFF0000,
            "green": 0x00FF00,
            "blue": 0x0000FF,
            "yellow": 0xFFFF00,
            "purple": 0xFF00FF,
            "cyan": 0x00FFFF,
            "white": 0xFFFFFF,
            "black": 0x000000,
            "gray": 0xFF0000,
        }
        try:
            for entity_id in self._entities:
                color = color_mapping[self._entities[entity_id].color]
                # print(f"Setting led color of entity {entity_id} to {color}")
                # color = color_mapping["black"]
                self.set_ledup(entity_id, color)
                self.set_leddown(entity_id, color)
        except KeyError as e:
            print("Color not found in color mapping")
            raise SyntaxError(e)

    def set_ledup(self, entity_id, led_colors):
        json_msg = {
            "cmd_type": "ledup",
            "args_length": 6,
            "args": {
                "0": led_colors,
                "1": 14,
                "2": led_colors,
                "3": 14,
                "4": led_colors,
                "5": 14,
            },
        }
        json_str = json.dumps(json_msg)
        self.mqtt_client.publish(
            f"/VSWARM{entity_id}_robot/cmd", json_str.encode("utf-8")
        )

    def set_leddown(self, entity_id, led_colors):
        json_msg = {
            "cmd_type": "leddown",
            "args_length": 6,
            "args": {
                "0": led_colors,
                "1": 30,
                "2": led_colors,
                "3": 30,
                "4": led_colors,
                "5": 30,
            },
        }
        json_str = json.dumps(json_msg)
        self.mqtt_client.publish(
            f"/VSWARM{entity_id}_robot/cmd", json_str.encode("utf-8")
        )
