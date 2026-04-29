import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from Box2D import b2Vec2
from modules.deployment.entity.base_entity import Entity
from modules.deployment.engine.box2d_engine import (
    Box2DEngine,
    PIDController,
)  # Replace 'your_module' with the module containing Box2DEngine


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


class TestBox2DEngine(unittest.TestCase):
    def setUp(self):
        self.engine = Box2DEngine(
            gravity=(0, -10)
        )  # Initialize the engine with gravity
        self.entity = MockEntity(
            1, initial_position=[0, 0], size=1.0, color="blue", movable=True
        )
        self.engine.add_entity(self.entity)

    @patch.object(Box2DEngine, "apply_force")
    def test_apply_force(self, mock_apply_force):
        # Apply a force to an entity and verify
        force = np.array([10.0, 0.0])
        self.engine.apply_force(self.entity.id, force)
        mock_apply_force.assert_called_once_with(self.entity.id, force)

    @patch.object(Box2DEngine, "control_velocity")
    def test_control_velocity(self, mock_control_velocity):
        # Test control_velocity by setting desired_velocity
        desired_velocity = np.array([5.0, 0.0])
        self.engine.control_velocity(self.entity.id, desired_velocity, dt=0.1)
        mock_control_velocity.assert_called_once_with(
            self.entity.id, desired_velocity, dt=0.1
        )

    def test_step_updates_position_and_velocity(self):
        # Test the step method updates positions and velocities
        initial_position, _ = self.engine.get_entity_state(self.entity.id)
        self.engine.step(1.0)  # Step by 1 second
        updated_position, updated_velocity = self.engine.get_entity_state(
            self.entity.id
        )
        self.assertFalse(
            np.array_equal(initial_position, updated_position),
            "Position should update after step",
        )

    def test_add_entity(self):
        # Test adding an entity creates a Box2D body
        self.assertIn(self.entity.id, self.engine.bodies)
        body = self.engine.bodies[self.entity.id]
        self.assertIsNotNone(body)
        self.assertEqual(body.position, b2Vec2(self.entity.position))

    def test_remove_entity(self):
        # Test removing an entity deletes it from Box2D world
        self.engine.remove_entity(self.entity.id)
        self.assertNotIn(self.entity.id, self.engine.bodies)

    def test_add_joint(self):
        # Test adding a joint between two entities
        entity2 = MockEntity(2, initial_position=[5, 0], size=1.0, color="red")
        self.engine.add_entity(entity2)
        self.engine.add_joint(self.entity.id, entity2.id, distance=5.0)
        self.assertIn((self.entity.id, entity2.id), self.engine.joints)

    def test_remove_joint(self):
        # Test removing a joint between two entities
        entity2 = MockEntity(2, initial_position=[5, 0], size=1.0, color="red")
        self.engine.add_entity(entity2)
        self.engine.add_joint(self.entity.id, entity2.id, distance=5.0)
        self.engine.remove_joint(self.entity.id, entity2.id)
        self.assertNotIn((self.entity.id, entity2.id), self.engine.joints)

    # Test PIDController compute method
    def test_pid_controller_compute(self):
        pid_controller = PIDController(1.0, 0.1, 0.05)
        setpoint = 10.0
        measurement = 8.0
        dt = 0.1

        # Call the compute method
        output = pid_controller.compute(setpoint, measurement, dt)

        # Check if the output is as expected
        expected_output = 3  # Calculated value based on given parameters
        self.assertAlmostEqual(output, expected_output, places=1)

    # Test PIDController set_parameters method
    def test_pid_controller_set_parameters(self):
        pid_controller = PIDController(1.0, 0.1, 0.05)

        # Set new parameters
        pid_controller.set_parameters(2.0, 0.2, 0.1)

        # Verify the parameters were updated
        self.assertEqual(pid_controller.kp, 2.0)
        self.assertEqual(pid_controller.ki, 0.2)
        self.assertEqual(pid_controller.kd, 0.1)

    def test_get_entities_state(self):
        # Add a second entity for testing
        entity2 = MockEntity(
            2, initial_position=[1, 1], size=1.0, color="red", movable=True
        )
        self.engine.add_entity(entity2)

        # Get states of all entities
        positions, velocities = self.engine.get_entities_state()

        # Verify that we have the correct number of entities
        self.assertEqual(positions.shape[0], 2)
        self.assertEqual(velocities.shape[0], 2)

    @patch.object(Box2DEngine, "get_entity_state")
    def test_apply_force(self, mock_get_entity_state):
        # Mock the entity state to return a specific value
        mock_get_entity_state.return_value = (
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        )
        force = np.array([10.0, 0.0])

        # Call apply_force
        self.engine.apply_force(self.entity.id, force)

        # Verify that the ApplyForceToCenter method was called
        body = self.engine.bodies[self.entity.id]
        body.ApplyForceToCenter = MagicMock()  # Mock the method to verify calls
        self.engine.apply_force(self.entity.id, force)

        body.ApplyForceToCenter.assert_called_once_with(b2Vec2(tuple(force)), True)

    @patch.object(Box2DEngine, "apply_force")
    def test_control_velocity(self, mock_apply_force):
        desired_velocity = np.array([5.0, 0.0])
        dt = 0.1

        # Call control_velocity
        self.engine.control_velocity(self.entity.id, desired_velocity, dt)

        # Assert that apply_force is called with the expected parameters
        mock_apply_force.assert_called_once()


if __name__ == "__main__":
    unittest.main()
