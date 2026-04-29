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

import numpy as np

from .base_engine import Engine
from modules.deployment.entity.base_entity import Entity
from modules.deployment.utils.quad_tree import QuadTree


class QuadTreeEngine(Engine):
    def __init__(
        self,
        world_size: tuple | list | np.ndarray,
        damping=0.95,
        alpha=0.7,
        collision_check=True,
        joint_constraint=True,
    ):
        """
        Physics engine that uses a quad tree for collision detection.
        Args:
            world_size(width,height): The size of the world in which the entities exist.
            damping: The damping factor for velocity.
            alpha: The alpha value for low-pass filter.
            collision_check: Whether to perform collision checks.
            joint_constraint: Whether to apply joint constraints.
        """
        super().__init__()
        self.world_size = np.array(world_size)
        self.quad_tree = QuadTree(
            -world_size[0] * 0.5,
            -world_size[1] * 0.5,
            world_size[0] * 0.5,
            world_size[1] * 0.5,
        )
        self._damping = damping
        self._alpha = alpha
        self._collision_check = collision_check
        self._joint_constraint = joint_constraint

    def add_entity(self, entity: Entity):
        """
        Add an entity to the environment.
        Args:
            entity (Entity): The entity to add.
        """
        super().add_entity(entity)
        self.quad_tree.insert(entity)

    def remove_entity(self, entity_id: int):
        """
        Remove an entity from the environment.
        Args:
            entity_id (int): The unique ID of the entity to remove.
        """

        entity = self._entities[entity_id]
        super().remove_entity(entity_id)

        joints_copy = self._joints.copy()

        for entity_id1, entity_id2 in joints_copy.keys():
            if entity_id == entity_id1 or entity_id == entity_id2:
                print(f"Removing joint between {entity_id1} and {entity_id2}")
                self.remove_joint(entity_id1, entity_id2)

        self.quad_tree.remove(entity)

    def set_position(self, entity_id: int, position: np.ndarray):
        """
        Set the position of an entity in the environment.
        Args:
            entity_id (int): The unique ID of the entity.
            position (np.ndarray): The new position of the entity.
        """
        super().set_position(entity_id, position)
        entity = self._entities[entity_id]
        self.quad_tree.update(entity)

    def step(self, delta_time: float):
        """
        Perform a physics step in the environment.
        """
        for entity in self._entities.values():
            entity.velocity *= self._damping  # Apply damping
            if self._collision_check:
                possible_collisions = self.quad_tree.retrieve(entity)
                for other in possible_collisions:
                    if entity.id != other.id and self._check_collision(entity, other):
                        dv1, dv2 = self._resolve_collision(entity, other)
                        entity.velocity += dv1
                        other.velocity += dv2
            velocity_adjustment = self._adjust_velocity_near_boundary(entity)
            entity.velocity += velocity_adjustment
        if self._joint_constraint:
            joints_copy = self._joints.copy()
            for (entity_id1, entity_id2), desired_length in joints_copy.items():
                entity1 = self._entities[entity_id1]
                entity2 = self._entities[entity_id2]
                dv1, dv2 = self._resolve_joint(entity1, entity2, desired_length)
                entity1.velocity += dv1
                entity2.velocity += dv2

        for entity in self._entities.values():
            if entity.moveable:
                # Update position based on velocity and delta_time
                entity.position += entity.velocity * delta_time
                # leave 0.05 margin to avoid entities getting stuck at the edge
                # entity.position = np.clip(entity.position, -0.5 * self.world_size, 0.5 * self.world_size)
                self.set_position(entity.id, entity.position)
        # Resolve overlaps

    def _adjust_velocity_near_boundary(self, entity: Entity) -> np.ndarray:
        """
        Adjust the velocity of an entity when it is near the boundary.
        Args:
            entity (Entity): The entity to adjust velocity for.
        Returns:
            np.ndarray: The velocity adjustment to be applied.
        """
        velocity_adjustment = np.zeros(2)
        margin = 0.1  # Velocity adjustment starts when entity is within 0.5 meters of the boundary
        adjustment_coefficient = 10  # Strength of the velocity adjustment

        # Calculate the distance to each boundary
        distances = np.array(
            [
                entity.position[0]
                - (-0.5 * self.world_size[0]),  # Distance to left boundary
                0.5 * self.world_size[0]
                - entity.position[0],  # Distance to right boundary
                entity.position[1]
                - (-0.5 * self.world_size[1]),  # Distance to bottom boundary
                0.5 * self.world_size[1]
                - entity.position[1],  # Distance to top boundary
            ]
        )

        # Adjust velocity based on proximity to the boundaries
        if distances[0] < margin:  # Near left boundary
            velocity_adjustment[0] += adjustment_coefficient * (margin - distances[0])
        elif distances[1] < margin:  # Near right boundary
            velocity_adjustment[0] -= adjustment_coefficient * (margin - distances[1])

        if distances[2] < margin:  # Near bottom boundary
            velocity_adjustment[1] += adjustment_coefficient * (margin - distances[2])
        elif distances[3] < margin:  # Near top boundary
            velocity_adjustment[1] -= adjustment_coefficient * (margin - distances[3])

        return velocity_adjustment

    def _resolve_overlaps(self):
        """
        Resolve overlaps between entities after movement.
        Returns:
            bool: True if any overlaps were resolved, False otherwise.
        """
        resolved_any = False
        for entity in self._entities.values():
            possible_collisions = self.quad_tree.retrieve(entity)
            for other in possible_collisions:
                if entity.id != other.id and self._check_collision(entity, other):
                    if not (entity.moveable and other.moveable):
                        continue
                    self._resolve_overlap(entity, other)
                    resolved_any = True
        return resolved_any

    def _resolve_overlap(self, entity1: Entity, entity2: Entity):
        """
        Resolve overlap between two entities by adjusting their positions.
        """

        overlap_vector = entity1.position - entity2.position
        overlap_distance = np.linalg.norm(overlap_vector)
        if overlap_distance == 0:
            overlap_vector = np.array([0.00001, 0])
        correction_vector = (
            overlap_vector
            * (entity1.size + entity2.size - overlap_distance)
            / overlap_distance
        )
        entity1.position += correction_vector / 2
        entity2.position -= correction_vector / 2
        self.set_position(entity1.id, entity1.position)
        self.set_position(entity2.id, entity2.position)

    def apply_force(self, entity_id: int, force: np.ndarray):
        """
        Apply a force to an entity in the environment.
        """
        if entity_id not in self._entities:
            raise ValueError("Entity does not exist in the environment.")
        entity = self._entities[entity_id]
        acceleration = force / entity.mass
        entity.velocity += acceleration

    def control_velocity(self, entity_id: int, desired_velocity: np.ndarray, dt=None):
        """
        Control the velocity of an entity in the environment with damping effect
        """
        if entity_id not in self._entities:
            raise ValueError("Entity does not exist in the environment.")

        entity = self._entities[entity_id]
        current_velocity = entity.velocity

        # Apply low-pass filter to update the velocity
        new_velocity = (
            self._alpha * desired_velocity + (1 - self._alpha) * current_velocity
        )

        # Update the entity's velocity
        self.set_velocity(entity_id, new_velocity)

    @staticmethod
    def _check_collision(entity1: Entity, entity2: Entity) -> bool:
        """
        Check if two entities are colliding.
        """
        distance = np.linalg.norm(entity1.position - entity2.position)
        return distance < (entity1.size + entity2.size)

    @staticmethod
    def _resolve_collision(entity1: Entity, entity2: Entity):
        """
        Resolve a collision between two entities.
        Returns:
            dv1 (np.ndarray): Change in velocity for entity1.
            dv2 (np.ndarray): Change in velocity for entity2.
        """
        collision_vector = entity1.position - entity2.position
        if np.linalg.norm(collision_vector) == 0:
            collision_vector = np.array([0.00001, 0])
        collision_normal = collision_vector / np.linalg.norm(collision_vector)
        relative_velocity = entity1.velocity - entity2.velocity
        velocity_along_normal = np.dot(relative_velocity, collision_normal)

        if velocity_along_normal > 0:
            return np.zeros_like(entity1.velocity), np.zeros_like(entity2.velocity)

        if entity1.moveable and entity2.moveable:
            impulse = (2 * velocity_along_normal) / (entity1.mass + entity2.mass)
            dv1 = -impulse * entity2.mass * collision_normal
            dv2 = impulse * entity1.mass * collision_normal
        elif entity1.moveable and not entity2.moveable:
            dv1 = -2 * velocity_along_normal * collision_normal
            dv2 = np.zeros_like(entity2.velocity)
        elif not entity1.moveable and entity2.moveable:
            dv1 = np.zeros_like(entity1.velocity)
            dv2 = 2 * velocity_along_normal * collision_normal
        else:
            dv1 = np.zeros_like(entity1.velocity)
            dv2 = np.zeros_like(entity2.velocity)

        return dv1, dv2

    @staticmethod
    def _resolve_joint(entity1: Entity, entity2: Entity, desired_length: float):
        """
        Resolve a joint constraint between two entities.
        Returns:
            dv1 (np.ndarray): Change in velocity for entity1.
            dv2 (np.ndarray): Change in velocity for entity2.
        """
        joint_vector = entity1.position - entity2.position
        joint_length = np.linalg.norm(joint_vector)

        correction_vector = (
            joint_vector * (desired_length - joint_length) / joint_length
        )
        correction_velocity = (
            correction_vector / 2
        )  # Apply half the correction to each entity

        dv1 = correction_velocity
        dv2 = -correction_velocity

        return dv1, dv2
