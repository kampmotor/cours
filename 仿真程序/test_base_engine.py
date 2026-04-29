import unittest
import numpy as np
from modules.deployment.entity.base_entity import (
    Entity,
)  # Make sure to import your Entity class here
from modules.deployment.engine.base_engine import Engine


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


class MockEngine(Engine):
    def step(self, delta_time: float):
        # A simple implementation that does nothing for the purpose of testing
        pass

    def apply_force(self, entity_id: int, force: np.ndarray):
        pass

    def control_velocity(self, entity_id: int, desired_velocity: np.ndarray, dt=None):
        pass


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.engine = MockEngine()  # Create an instance of the MockEngine
        self.entity1 = MockEntity(1, movable=True)
        self.entity2 = MockEntity(2, movable=True)

    def test_add_entity(self):
        self.engine.add_entity(self.entity1)
        self.assertIn(self.entity1.id, self.engine._entities)

    def test_add_entity_duplicate(self):
        self.engine.add_entity(self.entity1)
        with self.assertRaises(ValueError):
            self.engine.add_entity(self.entity1)

    def test_remove_entity(self):
        self.engine.add_entity(self.entity1)
        self.engine.remove_entity(self.entity1.id)
        self.assertNotIn(self.entity1.id, self.engine._entities)

    def test_remove_non_existent_entity(self):
        with self.assertRaises(ValueError):
            self.engine.remove_entity(99)

    def test_add_joint(self):
        self.engine.add_entity(self.entity1)
        self.engine.add_entity(self.entity2)
        self.engine.add_joint(self.entity1.id, self.entity2.id, distance=5.0)
        self.assertIn((self.entity1.id, self.entity2.id), self.engine._joints)

    def test_add_joint_same_entity(self):
        self.engine.add_entity(self.entity1)
        with self.assertRaises(ValueError):
            self.engine.add_joint(self.entity1.id, self.entity1.id, distance=5.0)

    def test_add_joint_existed_entity(self):
        self.engine.add_joint(self.entity1.id, self.entity2.id, distance=5.0)
        with self.assertRaises(ValueError):
            self.engine.add_joint(self.entity1.id, self.entity2.id, distance=5.0)

    def test_remove_joint(self):
        self.engine.add_entity(self.entity1)
        self.engine.add_entity(self.entity2)
        self.engine.add_joint(self.entity1.id, self.entity2.id, distance=5.0)
        self.engine.remove_joint(self.entity1.id, self.entity2.id)
        self.assertNotIn((self.entity1.id, self.entity2.id), self.engine._joints)

    def test_remove_non_existent_joint(self):
        with self.assertRaises(ValueError):
            self.engine.remove_joint(1, 2)

    def test_set_position(self):
        self.engine.add_entity(self.entity1)
        new_position = np.array([1.0, 2.0])
        self.engine.set_position(self.entity1.id, new_position)
        self.assertTrue(np.array_equal(self.entity1.position, new_position))

    def test_set_position_non_existent_entity(self):
        with self.assertRaises(ValueError):
            self.engine.set_position(99, np.array([1.0, 2.0]))

    def test_get_entities_state(self):
        self.engine.add_entity(self.entity1)
        self.engine.add_entity(self.entity2)
        positions, velocities = self.engine.get_entities_state()
        self.assertEqual(positions.shape[0], 2)
        self.assertEqual(velocities.shape[0], 2)

    def test_clear_entities(self):
        self.engine.add_entity(self.entity1)
        self.engine.clear_entities()
        self.assertEqual(len(self.engine._entities), 0)
        self.assertEqual(len(self.engine._joints), 0)

    def test_get_entity_state(self):
        self.engine.add_entity(self.entity1)
        position, velocity = self.engine.get_entity_state(self.entity1.id)
        self.assertTrue(np.array_equal(position, self.entity1.position))
        self.assertTrue(np.array_equal(velocity, self.entity1.velocity))

    def test_get_entity_state_non_existent(self):
        with self.assertRaises(ValueError):
            self.engine.get_entity_state(99)

    # def test_apply_force(self):
    #     self.engine.add_entity(self.entity1)
    #     initial_velocity = self.entity1.velocity.copy()
    #     force = np.array([1.0, 0.0])
    #     self.engine.apply_force(self.entity1.id, force)
    #     expected_velocity = initial_velocity + force
    #     self.assertTrue(np.array_equal(self.entity1.velocity, expected_velocity))

    # def test_apply_force_non_existent_entity(self):
    #     with self.assertRaises(ValueError):
    #         self.engine.apply_force(99, np.array([1.0, 0.0]))

    # def test_control_velocity(self):
    #     self.engine.add_entity(self.entity1)
    #     desired_velocity = np.array([2.0, 3.0])
    #     self.engine.control_velocity(self.entity1.id, desired_velocity)
    #     self.assertTrue(np.array_equal(self.entity1.velocity, desired_velocity))

    # def test_control_velocity_non_existent_entity(self):
    #     with self.assertRaises(ValueError):
    #         self.engine.control_velocity(99, np.array([1.0, 0.0]))

    def test_set_yaw_existent_entity(self):
        self.engine.add_entity(self.entity1)
        entity_id = self.entity1.id
        desired_velocity = 45.0
        self.engine.set_yaw(entity_id, desired_velocity)
        self.assertEqual(self.engine._entities[entity_id].yaw, desired_velocity)

    def test_set_yaw_non_existent_entity(self):
        with self.assertRaises(ValueError):
            self.engine.set_yaw(99, 45.0)

    def test_set_velocity_existent_entity(self):
        self.engine.add_entity(self.entity1)
        entity_id = self.entity1.id
        desired_velocity = 2.0
        self.engine.set_velocity(entity_id, desired_velocity)
        self.assertEqual(self.engine._entities[entity_id].velocity, desired_velocity)

    def test_set_velocity_non_existent_entity(self):
        with self.assertRaises(ValueError):
            self.engine.set_velocity(99, 2.0)

    def test_abstract_method(self):
        self.engine = Engine()

        with self.assertRaises(NotImplementedError):
            self.engine.step(None)

        with self.assertRaises(NotImplementedError):
            self.engine.apply_force(None, None)

        with self.assertRaises(NotImplementedError):
            self.engine.control_velocity(None, None)


if __name__ == "__main__":
    unittest.main()
