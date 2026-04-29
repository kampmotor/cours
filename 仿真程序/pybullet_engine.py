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

import pybullet as p
import pybullet_data
import numpy as np
from .base_engine import Engine
from modules.deployment.entity.base_entity import Entity


class PyBullet2DEngine(Engine):
    def __init__(
        self,
        world_size: tuple | list | np.ndarray,
        damping=0.95,
        alpha=0.7,
        collision_check=False,
        joint_constraint=True,
    ):
        """
        PyBullet-based physics engine for 2D simulations, avoiding direct use of Entity objects.
        Args:
            world_size (width, height): The size of the world.
            damping: Damping factor for velocity.
            alpha: Low-pass filter factor for velocity control.
            collision_check: Whether to enable collision checks.
            joint_constraint: Whether to enable joint constraints.
        """
        super().__init__()
        self.joints_map = {}
        self.world_size = np.array(world_size)
        self._damping = damping
        self._alpha = alpha
        self._collision_check = collision_check
        self._joint_constraint = joint_constraint

        # Initialize PyBullet physics client (use GUI for visualization)
        self.physics_client = p.connect(p.DIRECT)  # Change to p.GUI for GUI mode
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )  # Set PyBullet's data path
        p.setGravity(0, 0, -0.0)  # Simulate gravity for 2D environment
        p.setTimeStep(1 / 10)

        # Store mapping of entity_id to PyBullet body_id and other related data
        self.entity_map = {}

    def add_entity(self, entity: Entity):
        """
        Add an entity to the PyBullet simulation as a 2D object.
        """
        super().add_entity(entity)

        # Determine the shape (circle or rectangle) based on entity's size
        if isinstance(entity.size, float):  # Circle
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=entity.size, height=0.01
            )
        else:  # Rectangle
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[entity.size[0] / 2, entity.size[1] / 2, 0.01]
            )

        mass = entity.mass if entity.moveable else 0  # Set mass 0 if not moveable

        # Disable collision if self._collision_check is False by using a flag in createMultiBody
        if self._collision_check:
            body_id = p.createMultiBody(
                mass,
                collision_shape,
                -1,
                basePosition=[entity.position[0], entity.position[1], 0],
            )
        else:
            # Use a "ghost" object without collisions (zero mass and disable collision response)
            body_id = p.createMultiBody(
                mass,
                collision_shape,
                -1,
                basePosition=[entity.position[0], entity.position[1], 0],
            )
            p.setCollisionFilterGroupMask(
                body_id, -1, 0, 0
            )  # Disable collision for this body

        # Store the entity's data in the entity_map
        self.entity_map[entity.id] = {
            "body_id": body_id,
            "position": np.array(entity.position),
            "velocity": np.array(entity.velocity),
            "size": entity.size,
            "mass": entity.mass,
            "moveable": entity.moveable,
        }

    def remove_entity(self, entity_id: int):
        """
        Remove an entity from the PyBullet simulation and internal data structure.
        """
        if entity_id in self.entity_map:
            body_id = self.entity_map[entity_id]["body_id"]
            p.removeBody(body_id)  # Remove from PyBullet simulation
            del self.entity_map[entity_id]  # Remove from internal storage
            super().remove_entity(entity_id)
        else:
            raise ValueError(f"Entity {entity_id} does not exist in the environment.")

    def step(self, delta_time: float):
        """
        Perform a physics step in the PyBullet simulation.
        """
        # Only perform simulation steps in PyBullet; no manual collision/joint handling required
        p.stepSimulation()

        # Update positions and velocities for all entities
        for entity_id, data in self.entity_map.items():
            if isinstance(data, dict) and data["moveable"]:
                position, _ = p.getBasePositionAndOrientation(data["body_id"])
                velocity, _ = p.getBaseVelocity(data["body_id"])

                # Update position and velocity in internal map
                self.entity_map[entity_id]["position"] = np.array(position[:2])
                self.entity_map[entity_id]["velocity"] = np.array(velocity[:2])
                self._entities[entity_id].position = np.array(position[:2])
                self._entities[entity_id].velocity = np.array(velocity[:2])

    def apply_force(self, entity_id: int, force: np.ndarray):
        """
        Apply a force to an entity in the PyBullet simulation.
        """
        if entity_id in self.entity_map:
            body_id = self.entity_map[entity_id]["body_id"]
            p.applyExternalForce(
                body_id, -1, [force[0], force[1], 0], [0, 0, 0], p.WORLD_FRAME
            )
        else:
            raise ValueError(f"Entity {entity_id} does not exist in the environment.")

    def control_velocity(self, entity_id: int, desired_velocity: np.ndarray, dt=None):
        """
        Control the velocity of an entity in the PyBullet simulation using a low-pass filter.
        Instead of directly resetting the velocity, we apply forces to modify the velocity,
        which preserves the collision behavior.
        """
        if entity_id in self.entity_map:
            body_id = self.entity_map[entity_id]["body_id"]
            current_velocity = np.array(
                p.getBaseVelocity(body_id)[0][:2]
            )  # Get current 2D velocity

            # Apply low-pass filter to smooth the velocity control
            new_velocity = (
                self._alpha * desired_velocity + (1 - self._alpha) * current_velocity
            )

            # Calculate the difference between the current velocity and desired velocity
            velocity_diff = new_velocity - current_velocity

            # Calculate the mass of the entity to compute the necessary force
            mass = self.entity_map[entity_id]["mass"]

            if mass > 0:  # Only apply force to moveable objects
                # Use F = m * a (acceleration = velocity difference / time step)
                force = (
                    mass * velocity_diff / (dt if dt else 1 / 10)
                )  # Assuming 10 Hz time step

                # Apply the calculated force to achieve the desired velocity
                p.applyExternalForce(
                    body_id, -1, [force[0], force[1], 0], [0, 0, 0], p.WORLD_FRAME
                )

            # Update damping (this will slow down the entity over time)
            p.changeDynamics(body_id, -1, linearDamping=self._damping)
        else:
            raise ValueError(f"Entity {entity_id} does not exist in the environment.")

    def add_joint(self, entity_id1: int, entity_id2: int, distance: float):
        """
        Add a joint between two entities in the environment.
        """
        # Only add joints if self._joint_constraint is True
        if self._joint_constraint:
            if entity_id1 in self.entity_map and entity_id2 in self.entity_map:
                body_id1 = self.entity_map[entity_id1]["body_id"]
                body_id2 = self.entity_map[entity_id2]["body_id"]
                self._entities[entity_id1].position = np.array(
                    p.getBasePositionAndOrientation(body_id1)[0][:2]
                )
                self._entities[entity_id2].position = np.array(
                    p.getBasePositionAndOrientation(body_id2)[0][:2]
                )
                dx = (
                    self._entities[entity_id1].position[0]
                    - self._entities[entity_id2].position[0]
                )
                dy = (
                    self._entities[entity_id1].position[1]
                    - self._entities[entity_id2].position[1]
                )

                constraint_id = p.createConstraint(
                    body_id1,
                    -1,
                    body_id2,
                    -1,
                    p.JOINT_POINT2POINT,
                    [0, 0, 0],
                    [0, 0, 0],
                    [dx, dy, 0],
                )
                # Store the joint in a separate map for later removal
                self.joints_map[(entity_id1, entity_id2)] = constraint_id
                print(f"Added joint between entities {entity_id1} and {entity_id2}.")
            else:
                raise ValueError(
                    f"One or both entities ({entity_id1}, {entity_id2}) do not exist in the environment."
                )

    def remove_joint(self, entity_id1: int, entity_id2: int):
        """
        Remove a joint between two entities in the environment.
        Args:
            entity_id1 (int): The unique ID of the first entity.
            entity_id2 (int): The unique ID of the second entity.
        """
        if (entity_id1, entity_id2) in self.joints_map:
            constraint_id = self.joints_map.pop((entity_id1, entity_id2))
            p.removeConstraint(constraint_id)
        else:
            raise ValueError(
                f"No joint exists between entities {entity_id1} and {entity_id2}."
            )

    def set_position(self, entity_id: int, position: np.ndarray):
        """
        Set the position of an entity in the environment.
        Args:
            entity_id (int): The unique ID of the entity.
            position (np.ndarray): The new position of the entity.
        """
        if entity_id in self.entity_map:
            body_id = self.entity_map[entity_id]["body_id"]
            p.resetBasePositionAndOrientation(
                body_id, [position[0], position[1], 0], [0, 0, 0, 1]
            )
            self.entity_map[entity_id]["position"] = np.array(position)
        else:
            raise ValueError(f"Entity {entity_id} does not exist in the environment.")

    def set_velocity(self, entity_id: int, velocity: np.ndarray):
        """
        Set the velocity of an entity in the environment.
        Args:
            entity_id (int): The unique ID of the entity.
            velocity (np.ndarray): The new velocity of the entity.
        """
        if entity_id in self.entity_map:
            body_id = self.entity_map[entity_id]["body_id"]
            p.resetBaseVelocity(body_id, linearVelocity=[velocity[0], velocity[1], 0])
            self.entity_map[entity_id]["velocity"] = np.array(velocity)
        else:
            raise ValueError(f"Entity {entity_id} does not exist in the environment.")

    def get_entities_state(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the state of all entities in the environment.
        Returns:
            positions (np.ndarray): The positions of all entities.
            velocities (np.ndarray): The velocities of all entities.
        """
        positions = []
        velocities = []
        for entity_data in self.entity_map.values():
            if isinstance(
                entity_data, dict
            ):  # Only process valid entity data (skip joints)
                positions.append(entity_data["position"])
                velocities.append(entity_data["velocity"])
        return np.array(positions), np.array(velocities)

    def get_entity_state(self, entity_id: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the state of a specific entity in the environment.
        Args:
            entity_id (int): The unique ID of the entity.
        Returns:
            position (np.ndarray): The position of the entity.
            velocity (np.ndarray): The velocity of the entity.
        """
        if entity_id in self.entity_map:
            position = self.entity_map[entity_id]["position"]
            velocity = self.entity_map[entity_id]["velocity"]
            return position, velocity
        else:
            raise ValueError(f"Entity {entity_id} does not exist in the environment.")

    def clear_entities(self):
        """
        Clear all entities from the environment.
        """
        for entity_id, data in self.entity_map.items():
            if isinstance(data, dict):  # Only remove entities, not joints
                p.removeBody(data["body_id"])
        self.entity_map.clear()
        super().clear_entities()

    def __del__(self):
        """
        Clean up the PyBullet simulation on exit.
        """
        p.disconnect()
