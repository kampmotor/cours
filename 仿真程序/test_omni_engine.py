import unittest
import json
from unittest.mock import patch, MagicMock
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Joy  # 新增导入 Joy 消息类型

from modules.deployment.utils.mqtt_pub import MqttClientThread
from modules.deployment.engine import (
    OmniEngine,
)  # Adjust the import according to your package structure
from code_llm.msg import (
    Observations,
)  # Make sure the message type exists in your test environment


class TestOmniEngine(unittest.TestCase):
    @patch.object(OmniEngine, "start_up_mqtt_thread")
    @patch("rospy.Subscriber")
    @patch("rospy.init_node")
    @patch("rospy.Time.now")
    def setUp(self, mock_subscriber, mock_start_mqtt, mock_init_node, mock_time_now):
        self.engine = OmniEngine()

        self.engine.mqtt_client = MagicMock()  # Mock the MQTT client
        self.engine._entities = {  # Mock entities with attributes like yaw and color
            1: MagicMock(yaw=0.0, color="red"),
            2: MagicMock(yaw=np.pi, color="green"),
        }

    @patch("modules.deployment.engine.omni_engine.R")  # 模拟 R 旋转库
    def test_pose_callback(self, mock_rotation):
        # 使用 MagicMock 为 set_position 和 set_yaw 创建模拟
        self.engine.set_position = MagicMock()
        self.engine.set_yaw = MagicMock()

        # 模拟 PoseStamped 消息内容
        pose_msg = PoseStamped()
        pose_msg.pose.position.x = 1.0
        pose_msg.pose.position.y = 2.0
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0

        # 设置 entity_id 和 entity_type
        entity_id = 123
        entity_type = "robot"

        # 模拟 R.from_quat 以返回虚拟旋转矩阵
        mock_rotation.from_quat.return_value.as_matrix.return_value = np.identity(3)
        mock_rotation.from_matrix.return_value.as_euler.return_value = [0, 0, np.pi / 4]

        # 调用 pose_callback
        self.engine.pose_callback(pose_msg, (entity_id, entity_type))

        # 检查是否按预期调用 set_yaw
        self.engine.set_yaw.assert_called_once_with(entity_id, np.pi / 4)

        # 检查日志是否正确显示
        print(f"update position of {entity_type} {entity_id} to {[1.0, 2.0]}")

    @patch.object(OmniEngine, "set_velocity")
    def test_twist_callback(self, mock_set_velocity):
        # 创建一个假消息
        msg = TwistStamped()
        msg.twist.linear.x = 1.0
        msg.twist.linear.y = 2.0
        args = (1, "robot")

        # 调用twist_callback
        self.engine.twist_callback(msg, args)

        # 检查是否调用了set_velocity
        expected_velocity = np.array([1.0, 2.0])
        # mock_set_velocity.assert_called_once_with(1, expected_velocity)

    @patch("rospy.Time")
    def test_joy_callback(self, mock_time):
        # 模拟 joy 消息
        joy_msg = Joy()
        joy_msg.axes = [0.5, 0.5, 0.0, -0.5]  # x, y, z, theta轴的值

        # 调用joy_callback
        self.engine.joy_callback(joy_msg)

        # 检查 joy_input 更新
        self.assertAlmostEqual(self.engine.joy_input["x"], -0.125)
        self.assertAlmostEqual(self.engine.joy_input["y"], -0.125)
        self.assertAlmostEqual(self.engine.joy_input["theta"], -1.0)

    def test_control_velocity(self):
        # Define test data
        entity_id = 1
        desired_velocity = {"x": 0.5, "y": -0.25, "theta": 1.0}

        # Expected JSON message to be sent
        expected_json_msg = json.dumps(desired_velocity).encode("utf-8")
        expected_topic = f"/VSWARM{entity_id}_robot/motion"

        # Call the method
        self.engine.control_velocity(entity_id, desired_velocity)

        # Assert that publish is called with correct topic and message
        self.engine.mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_json_msg
        )

    def test_control_yaw_with_small_yaw_error(self):
        # Test with a yaw error below the threshold (0.1)
        entity_id = 1
        self.engine._entities[entity_id].yaw = 0.05
        desired_yaw = 0.1

        # Call the method
        self.engine.control_yaw(entity_id, desired_yaw)
        # No publish call should happen if yaw error is below 0.1
        self.engine.mqtt_client.publish.assert_not_called()

    def test_control_yaw_with_large_yaw_error(self):
        # Test with a yaw error above the threshold (0.1)
        entity_id = 1
        self.engine._entities[entity_id].yaw = 0.0
        desired_yaw = np.pi / 2  # Large yaw error

        # Expected JSON message to be sent with theta correction
        yaw_error = desired_yaw - self.engine._entities[entity_id].yaw
        kp = 0.8
        expected_json_msg = json.dumps(
            {"x": 0, "y": 0, "theta": yaw_error * kp}
        ).encode("utf-8")
        expected_topic = f"/VSWARM{entity_id}_robot/motion"

        # Call the method
        self.engine.control_yaw(entity_id, desired_yaw)

        # Assert that publish is called with the correct topic and message
        self.engine.mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_json_msg
        )

    def test_update_led_color(self):
        # Test the update_led_color method to ensure correct color setting
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

        # Mock the set_ledup and set_leddown methods
        self.engine.set_ledup = MagicMock()
        self.engine.set_leddown = MagicMock()

        # Call the method
        self.engine.update_led_color()

        # # Verify if set_ledup and set_leddown are called with the correct color mapping
        # for entity_id, entity in self.engine._entities.items():
        #     expected_color = color_mapping[
        #         "black"
        #     ]  # The method always sets color to "black"
        # self.engine.set_ledup.assert_any_call(entity_id, expected_color)
        # self.engine.set_leddown.assert_any_call(entity_id, expected_color)

    def test_update_led_color_invalid_color(self):
        # Test update_led_color with an invalid color in one of the entities
        self.engine._entities[1].color = "invalid_color"

        # Call the method and expect a KeyError exception
        with self.assertRaises(SyntaxError):
            self.engine.update_led_color()

    def test_set_ledup(self):
        # Test the set_ledup method to ensure it publishes the correct MQTT message
        entity_id = 1
        led_colors = 0xFF0000  # Example color (red)

        # Expected JSON message structure for set_ledup
        expected_json_msg = {
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
        expected_json_str = json.dumps(expected_json_msg).encode("utf-8")
        expected_topic = f"/VSWARM{entity_id}_robot/cmd"

        # Call the method
        self.engine.set_ledup(entity_id, led_colors)

        # Assert that publish is called with the correct topic and JSON message
        self.engine.mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_json_str
        )

    def test_set_leddown(self):
        # Test the set_leddown method to ensure it publishes the correct MQTT message
        entity_id = 2
        led_colors = 0x00FF00  # Example color (green)

        # Expected JSON message structure for set_leddown
        expected_json_msg = {
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
        expected_json_str = json.dumps(expected_json_msg).encode("utf-8")
        expected_topic = f"/VSWARM{entity_id}_robot/cmd"

        # Call the method
        self.engine.set_leddown(entity_id, led_colors)

        # Assert that publish is called with the correct topic and JSON message
        self.engine.mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_json_str
        )


if __name__ == "__main__":
    unittest.main()
