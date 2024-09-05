from uuid import UUID, uuid4
from typing import Dict, Any, Tuple, List
from state_snapshot import StateSnapshot
from enums import ObjectType


class Item:
    def __init__(self, matrika, position: Tuple[int, int]):
        self.id = uuid4()
        self.x, self.y = position
        self.matrika = matrika
        self.type: ObjectType = ObjectType.ITEM
        self.color = (255, 255, 255)  # Default color: white
        self.marked_for_deletion = False
        self.expiration = False
        self.expiration_timer = 60  # Default 60 seconds
        self.update_interval = self.matrika.UPDATE_INTERVAL
        self.reward = 0.0
        self.energy = 0.0
        self.nutrition = 0.0
        self.collision = True  # New attribute for collision
        self.consumable = False
        self.synchronized_params: List[str] = self._calculate_synchronized_params()

    def _calculate_synchronized_params(self) -> List[str]:
        return [
            param for param, value in self.__dict__.items()
            if isinstance(value, (int, float, UUID)) or 
               (param.endswith('_ID') and value is None)
        ]

    def consume(self, organism):
        pass
    
    def update_state(self, state_snapshot: StateSnapshot) -> Dict[str, Any]:
        if self.expiration_timer > 0:
            self.expiration_timer -= self.update_interval
            if self.expiration_timer <= 0:
                self.marked_for_deletion = True
        
        external_state_change = {}
        return external_state_change
    
    def apply_state(self, old_state: StateSnapshot, new_state: StateSnapshot):
       pass

class Food(Item):
    def __init__(self, matrika, position: Tuple[int, int], energy=1.0, nutrition=0.1, color=(255, 255, 0), expiration_timer=60):
        super().__init__(matrika, position)
        self.color = color
        self.expiration = True
        self.expiration_timer = expiration_timer
        self.energy = energy
        self.nutrition = nutrition
        self.reward = energy + 5 * nutrition
        self.consumable = True

    def consume(self, organism):
        organism.energy += self.energy
        organism.nutrition += self.nutrition
        organism.current_reward += self.reward