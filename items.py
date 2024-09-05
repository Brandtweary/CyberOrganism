from uuid import UUID, uuid4
from typing import Dict, Any, Tuple, List, Type
from state_snapshot import StateSnapshot
from enums import ObjectType
from shared_resources import calculate_synchronized_params
import inspect

item_class_map = ()

class Item:
    def __init__(self, SimulationEngine, position: Tuple[int, int]):
        self.id = uuid4()
        self.x, self.y = position
        self.sim_engine = SimulationEngine
        self.type: ObjectType = ObjectType.ITEM
        self.color = (255, 255, 255)  # Default color: white
        self.marked_for_deletion = False
        self.expiration = False
        self.expiration_timer = 60  # Default 60 seconds
        self.update_interval = self.sim_engine.UPDATE_INTERVAL
        self.reward = 0.0
        self.energy = 0.0
        self.nutrition = 0.0
        self.collision = True  # New attribute for collision
        self.consumable = False
        self.synchronized_params: List[str] = calculate_synchronized_params(self)

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

def get_all_item_subclasses() -> Dict[str, Type[Item]]:
    item_classes = {}
    for name, obj in globals().items():
        if inspect.isclass(obj) and issubclass(obj, Item) and obj != Item:
            item_classes[name.lower()] = obj
    return item_classes

def get_item_class(item_type: str) -> Type[Item]:
    return item_class_map.get(item_type.lower(), Item)