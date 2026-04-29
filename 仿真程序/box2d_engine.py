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

from abc import ABC
import numpy as np
# from Box2D import (
#     b2World,
#     b2CircleShape,
#     b2PolygonShape,
#     b2_dynamicBody,
#     b2_staticBody,
#     b2Vec2,
#     b2DistanceJointDef,
#     b2Body,
#     b2Joint,
#     b2RevoluteJointDef,
# )
from .base_engine import Engine
from modules.deployment.entity.base_entity import Entity


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def compute(self, setpoint, measurement, dt):
        error = setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def set_parameters(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd


# TODO： Fix the code below
class Box2DEngine(Engine, ABC):
    def __init__(self):
        pass
#     """
#     Physics engine that uses Box2D for physics simulation.
#     """
#
#     def __init__(self, gravity=(0, 0)):
#         super().__init__()
#         self.world = b2World(gravity=gravity)
#         self.bodies: dict[int, b2Body] = {}
#         self.joints: dict[tuple[int, int], b2Joint] = {}
#         self.velocity_controller = PIDController(0.005, 0, 0)
#
#     def add_entity(self, entity: Entity):
#         super().add_entity(entity)
#         if entity.shape == "circle":
#             shape = b2CircleShape(radius=entity.size)
#         else:
#             shape = b2PolygonShape(box=(entity.size[0] / 2, entity.size[1] / 2))
#
#         body_type = b2_dynamicBody if entity.moveable else b2_staticBody
#
#         body_def = {"position": entity.position, "type": body_type}
#
#         fixture_def = {
#             "shape": shape,
#             "density": entity.density,
#             "isSensor": not entity.collision,
#             "friction": 1.0,
#             "restitution": 0.0,
#         }
#
#         try:
#             body = self.world.CreateBody(**body_def)
#             body.CreateFixture(**fixture_def)
#             self.bodies[entity.id] = body
#         except Exception as e:
#             raise ValueError(f"Error adding entity {entity.id}: {e}")
#
#     def remove_entity(self, entity_id: int):
#         super().remove_entity(entity_id)
#         if entity_id in self.bodies:
#             self.world.DestroyBody(self.bodies[entity_id])
#             del self.bodies[entity_id]
#
#     def add_joint(self, entity_id1: int, entity_id2: int, distance: float = None):
#         super().add_joint(entity_id1, entity_id2, distance)
#         body1 = self.bodies.get(entity_id1)
#         body2 = self.bodies.get(entity_id2)
#
#         if body1 and body2:
#             if distance is None:
#                 distance = np.linalg.norm(body1.position - body2.position)
#
#             joint_def = b2DistanceJointDef(
#                 bodyA=body1,
#                 bodyB=body2,
#                 anchorA=body1.worldCenter,
#                 anchorB=body2.worldCenter,
#                 length=distance,
#                 frequencyHz=5,
#                 dampingRatio=100,
#             )
#
#             joint = self.world.CreateJoint(joint_def)
#             self.joints[(entity_id1, entity_id2)] = joint
#
#     def remove_joint(self, entity_id1: int, entity_id2: int):
#         super().remove_joint(entity_id1, entity_id2)
#         joint = self.joints.pop((entity_id1, entity_id2), None)
#         if joint:
#             self.world.DestroyJoint(joint)
#
#     def set_position(self, entity_id: int, position: np.ndarray):
#         super().set_position(entity_id, position)
#         body = self.bodies.get(entity_id)
#         if body:
#             body.position = position
#
#     def set_velocity(self, entity_id: int, velocity: np.ndarray):
#         super().set_velocity(entity_id, velocity)
#         body = self.bodies.get(entity_id)
#         if body:
#             body.linearVelocity = velocity
#
#     def get_entities_state(self):
#         positions = []
#         velocities = []
#         for entity_id in self._entities.keys():
#             position, velocity = self.get_entity_state(entity_id)
#             positions.append(position)
#             velocities.append(velocity)
#         return np.array(positions), np.array(velocities)
#
#     def get_entity_state(self, entity_id: int):
#         body = self.bodies.get(entity_id)
#         if body:
#             return np.array([body.position.x, body.position.y]), np.array(
#                 [body.linearVelocity.x, body.linearVelocity.y]
#             )
#         return super().get_entity_state(entity_id)
#
#     def step(self, delta_time: float):
#         self.world.Step(delta_time, 6, 2)
#         for entity_id in self._entities.keys():
#             position, velocity = self.get_entity_state(entity_id)
#             self.set_position(entity_id, position)
#             self.set_velocity(entity_id, velocity)
#
#     def apply_force(self, entity_id: int, force: np.ndarray):
#         body = self.bodies.get(entity_id)
#         print(f"Applying force {force} to entity {entity_id}")
#         if body:
#             body.ApplyForceToCenter(b2Vec2(tuple(force)), True)
#
#     def control_velocity(self, entity_id: int, desired_velocity: np.ndarray, dt=None):
#         body = self.bodies.get(entity_id)
#         print()
#         if body:
#             current_velocity = self.get_entity_state(entity_id)[1]
#             print(f"Current velocity: {current_velocity}")
#             force = self.velocity_controller.compute(
#                 desired_velocity, current_velocity, dt
#             )
#             self.apply_force(entity_id, force)
