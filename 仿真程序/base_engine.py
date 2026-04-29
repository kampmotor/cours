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

from abc import ABC, abstractmethod
import numpy as np
from modules.deployment.entity.base_entity import Entity


class Engine(ABC):
    """
    Base class for physics engines.
    """

    def __init__(self):
        self._entities: dict[int, Entity] = {}
        self._joints: dict[tuple[int, int], float] = {}

    def add_entity(self, entity: Entity):
        """
        Add an entity to the environment.
        Args:
            entity (Entity): The entity to add.
        """
        if entity.id in self._entities:
            raise ValueError(
                f"Entity_{entity.id} with the same ID already exists."
                f"Current entities: {list(self._entities.keys())}"
            )
        self._entities[entity.id] = entity

    def remove_entity(self, entity_id: int):
        """
        Remove an entity from the environment.
        Args:
            entity_id (int): The unique ID of the entity to remove.
        """
        if entity_id not in self._entities:
            raise ValueError("Entity does not exist in the environment.")
        self._entities.pop(entity_id)

    def add_joint(self, entity_id1: int, entity_id2: int, distance: float):
        """
        Add a joint between two entities in the environment.
        Args:
            entity_id1 (int): The unique ID of the first entity.
            entity_id2 (int): The unique ID of the second entity.
            distance (float): The distance between the two entities.
        """
        if entity_id1 == entity_id2:
            raise ValueError("Cannot create a joint between the same entity.")
        if (entity_id1, entity_id2) in self._joints.keys():
            raise ValueError("Joint already exists between the entities.")
        self._joints[(entity_id1, entity_id2)] = distance

    def remove_joint(self, entity_id1: int, entity_id2: int):
        """
        Remove a joint between two entities in the environment.
        Args:
            entity_id1 (int): The unique ID of the first entity.
            entity_id2 (int): The unique ID of the second entity.
        """
        if (entity_id1, entity_id2) not in self._joints.keys():
            raise ValueError("Joint does not exist between the entities.")
        self._joints.pop((entity_id1, entity_id2))

    def set_position(self, entity_id: int, position: np.ndarray):
        """
        Set the position of an entity in the environment.
        Args:
            entity_id (int): The unique ID of the entity.
            position (np.ndarray): The new position of the entity.
        """
        if entity_id not in self._entities.keys():
            raise ValueError("Entity does not exist in the environment.")
        self._entities[entity_id].position = position

    def set_yaw(self, entity_id: int, yaw: float):
        """
        Set the yaw of an entity in the environment.
        Args:
            entity_id (int): The unique ID of the entity.
            yaw (float): The new yaw of the entity.
        """
        if entity_id not in self._entities.keys():
            raise ValueError("Entity does not exist in the environment.")
        self._entities[entity_id].yaw = yaw

    def set_velocity(self, entity_id: int, velocity: np.ndarray):
        """
        Set the velocity of an entity in the environment.
        Args:
            entity_id (int): The unique ID of the entity.
            velocity (np.ndarray): The new velocity of the entity.
        """
        if entity_id not in self._entities.keys():
            raise ValueError("Entity does not exist in the environment.")
        self._entities[entity_id].velocity = velocity

    def get_entities_state(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the state of all entities in the environment.
        Returns:
            positions (np.ndarray): The positions of all entities.
            velocities (np.ndarray): The velocities of all entities.
        """
        entity_ids = list(self._entities.keys())
        positions = np.zeros((len(entity_ids), 2))
        velocities = np.zeros((len(entity_ids), 2))

        for idx, entity_id in enumerate(entity_ids):
            position, velocity = self.get_entity_state(entity_id)
            positions[idx] = position
            velocities[idx] = velocity
            return positions, velocities

    def get_entity_state(self, entity_id: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the state of a specific entity in the environment.
        Args:
            entity_id (int): The unique ID of the entity.
        Returns:
            position (np.ndarray): The position of the entity.
            velocity (np.ndarray): The velocity of the entity.
        """
        if entity_id not in self._entities.keys():
            raise ValueError("Entity does not exist in the environment.")
        return self._entities[entity_id].position, self._entities[entity_id].velocity

    def clear_entities(self):
        """
        Clear all entities from the environment.
        """
        self._entities.clear()
        self._joints.clear()

    def step(self, delta_time: float):
        """
        Perform a physics step in the environment.
        """
        raise NotImplementedError(
            "The step method must be implemented by the subclass."
        )

    def apply_force(self, entity_id: int, force):
        """
        Apply a force to an entity in the environment.
        """
        raise NotImplementedError(
            "The apply_force method must be implemented by the subclass."
        )

    def control_velocity(self, entity_id, desired_velocity, dt=None):
        """
        Control the velocity of an entity in the environment
        """
        raise NotImplementedError(
            "The control_velocity method must be implemented by the subclass."
        )
