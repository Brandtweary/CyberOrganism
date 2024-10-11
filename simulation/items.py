from uuid import UUID, uuid4
from typing import Dict, Any, Tuple, List, Type, Callable
from state_snapshot import StateSnapshot
from enums import ObjectType
import inspect
from functools import lru_cache

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
        self.synchronized_params: List[str] = []
        self.param_count: int = 0

    def consume(self, organism_state: Dict[str, Any]):
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

item_class_map: Dict[str, Callable[[], Type[Item]]] = {}

def register_item_class(name: str):
    def decorator(cls):
        item_class_map[name.lower()] = lambda: cls
        return cls
    return decorator

@lru_cache(maxsize=None)
def get_item_class(item_type: str) -> Type[Item]:
    class_getter = item_class_map.get(item_type.lower())
    if class_getter is None:
        raise ValueError(f"No item class found for type '{item_type}'")
    return class_getter()