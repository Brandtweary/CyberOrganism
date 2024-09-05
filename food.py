from items import Item
from typing import Tuple


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

    def consume(self, organism):
        organism.energy += self.energy
        organism.nutrition += self.nutrition
        organism.current_reward += self.reward