from items import Item, register_item_class
from typing import Tuple, Dict, Any

@register_item_class('food')
class Food(Item):
    def __init__(self, SimulationEngine, position: Tuple[int, int], energy=1.0, nutrition=0.1, color=(255, 255, 0), expiration_timer=60):
        super().__init__(SimulationEngine, position)
        self.color = color
        self.expiration = True
        self.expiration_timer = expiration_timer
        self.energy = energy
        self.nutrition = nutrition
        self.reward = energy + 5 * nutrition
        self.consumable = True

    def consume(self, organism_state: Dict[str, Any]):
        organism_state['energy'] += self.energy
        organism_state['nutrition'] += self.nutrition
        organism_state['current_reward'] += self.reward