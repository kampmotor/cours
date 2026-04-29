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

from modules.deployment.entity import Robot, Obstacle, Leader, Landmark, PushableObject
from modules.deployment.gymnasium_env.gymnasium_base_env import GymnasiumEnvironmentBase
from modules.deployment.utils.sample_point import *
from modules.deployment.engine import OmniEngine


class GymnasiumRealEnvironment(GymnasiumEnvironmentBase):
    def __init__(self, data_file: str = None):
        super().__init__(data_file=data_file)
        self.engine = OmniEngine()

    def init_entities(self):
        def add_specified_entities(entity_type, entity_class, color=None):
            for entity_data in self.data["entities"][entity_type]["specified"]:
                entity_position = np.array(entity_data["position"])
                entity = entity_class(
                    entity_data["id"], entity_position, entity_data["size"]
                )
                if color:
                    entity.color = entity_data.get("color", color)
                self.add_entity(entity)
                self.entities.append(entity)

        add_specified_entities("leader", Leader, "red")
        add_specified_entities("obstacle", Obstacle)
        add_specified_entities("landmark", Landmark)
        add_specified_entities("pushable_object", PushableObject)
        add_specified_entities("robot", Robot, "green")
