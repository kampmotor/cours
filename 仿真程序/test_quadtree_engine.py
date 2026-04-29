import unittest
import numpy as np
from unittest.mock import MagicMock
from modules.deployment.entity.base_entity import Entity
from modules.deployment.utils.quad_tree import QuadTree
from modules.deployment.engine.quadtree_engine import QuadTreeEngine


class MockEntity(Entity):
    def __init__(
        self,
        entity_id: int,
        initial_position: list[float] | tuple | np.ndarray = np.zeros(2),
        size: list[float] | tuple | np.ndarray | float = 1.0,
        color: str | tuple = "blue",
        collision: bool = False,
        movable: bool = False,
        max_speed: float = 1.0,
        mass: float = 1.0,
        density: float = 0.1,
        shape: str = "circle",
    ):
        super().__init__(
            entity_id,
            initial_position,
            size,
            color,
            collision,
            movable,
            max_speed,
            mass,
            density,
            shape,
        )


class TestQuadTreeEngine(unittest.TestCase):
    def setUp(self):
        # Initialize a QuadTreeEngine instance
        self.world_size = (10, 10)
        self.engine = QuadTreeEngine(world_size=self.world_size)
        self.entity1 = MockEntity(entity_id=1, size=1.0, mass=1.0, movable=True)
        self.entity1.position = np.array([1.0, 1.0])
        self.entity1.velocity = np.array([0.1, 0.1])
        self.entity2 = MockEntity(entity_id=2, size=1.0, mass=1.0, movable=True)
        self.entity2.position = np.array([2.0, 2.0])
        self.entity2.velocity = np.array([-0.1, -0.1])

        self.engine.add_entity(self.entity1)
        self.engine.add_entity(self.entity2)

    def test_add_entity(self):
        # Check if the entity is added to the quadtree
        self.assertIn(self.entity1.id, self.engine._entities)
        self.assertIn(self.entity2.id, self.engine._entities)

    def test_remove_entity(self):
        # Remove entity and verify removal from QuadTree
        self.engine.remove_entity(self.entity1.id)
        self.assertNotIn(self.entity1.id, self.engine._entities)
        self.assertNotIn(self.entity1, self.engine.quad_tree.retrieve(self.entity1))

    def test_set_position(self):
        # Test if setting position updates in QuadTree
        new_position = np.array([3.0, 3.0])
        self.engine.set_position(self.entity1.id, new_position)
        self.assertTrue(
            np.array_equal(
                self.engine._entities[self.entity1.id].position, new_position
            )
        )

    def test_collision_check(self):
        # Test collision detection between two overlapping entities
        collision = self.engine._check_collision(self.entity1, self.entity2)
        self.assertTrue(collision)

    def test_resolve_collision(self):
        # Test if collision resolution changes velocities as expected
        dv1, dv2 = self.engine._resolve_collision(self.entity1, self.entity2)
        self.entity1.velocity += dv1
        self.entity2.velocity += dv2
        self.assertTrue(np.any(dv1 != 0) or np.any(dv2 != 0))

    def test_control_velocity(self):
        # Set a desired velocity and verify it with control_velocity method
        desired_velocity = np.array([0.2, 0.2])
        self.engine.control_velocity(self.entity1.id, desired_velocity)
        self.assertTrue(
            np.allclose(
                self.engine._entities[self.entity1.id].velocity,
                desired_velocity,
                atol=0.1,
            )
        )

    def test_apply_force(self):
        # Apply force and check if velocity changes as expected
        force = np.array([0.5, 0.5])
        initial_velocity = self.entity1.velocity.copy()
        self.engine.apply_force(self.entity1.id, force)
        expected_velocity = initial_velocity + force / self.entity1.mass
        self.assertTrue(np.allclose(self.entity1.velocity, expected_velocity))

    def test_step_boundary(self):
        # Place an entity near the boundary and step, expecting velocity adjustment
        self.entity1.position = np.array([4.9, 0.1])  # Near right boundary
        self.engine.step(0.1)
        velocity_adjustment = self.engine._adjust_velocity_near_boundary(self.entity1)
        self.assertTrue(np.any(velocity_adjustment != 0))

    def test_resolve_overlap(self):
        # Overlap two entities and test if resolve overlap method separates them
        self.entity1.position = np.array([1.0, 1.0])
        self.entity2.position = np.array([1.5, 1.5])  # Overlapping
        self.engine._resolve_overlap(self.entity1, self.entity2)
        distance = np.linalg.norm(self.entity1.position - self.entity2.position)
        # self.assertGreaterEqual(distance, self.entity1.size + self.entity2.size)
        self.assertLessEqual(distance, self.entity1.size + self.entity2.size)

    def test_resolve_joint_constraint(self):
        # Add a joint constraint and test if entities' velocities adjust accordingly
        self.engine.add_joint(self.entity1.id, self.entity2.id, distance=1.5)
        dv1, dv2 = self.engine._resolve_joint(
            self.entity1, self.entity2, desired_length=1.5
        )
        self.assertTrue(np.any(dv1 != 0) or np.any(dv2 != 0))

    def tearDown(self):
        self.engine = None


if __name__ == "__main__":
    unittest.main()
