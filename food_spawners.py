import numpy as np
import time
from items import Item, register_item_class
from food import Food
import uuid

@register_item_class('food_spawner')
class FoodSpawner(Item):
    def __init__(self, SimulationEngine, position, spawn_frequency, regular_food_params, high_energy_food_params, spawn_range, entropy):
        super().__init__(SimulationEngine, position)
        self.spawn_frequency = spawn_frequency
        self.regular_food_params = regular_food_params
        self.high_energy_food_params = high_energy_food_params
        self.spawn_range = spawn_range
        self.entropy = entropy
        self.next_spawn_time = time.time()
        self.collision = False
        self.update_interval = 0
        self.expiration = False
        self.color = (0, 0, 0, 0)

    def calculate_spawn_delay(self):
        base_delay = 1 / self.spawn_frequency
        food_count = len([item for item in self.sim_engine.items if isinstance(item, Food)])
        
        if food_count <= 3:
            base_delay /= 2  # Double the spawn rate by halving the delay
        
        noise = np.random.normal(0, self.entropy * base_delay)
        return max(0.1, base_delay + noise)

    def should_spawn(self, current_time):
        # Check if it's time to spawn and if the maximum number of food items hasn't been reached
        return (current_time >= self.next_spawn_time and 
                len([item for item in self.sim_engine.items if isinstance(item, Food)]) < self.sim_engine.MAX_FOOD_ITEMS)

    def update_state(self, state_snapshot):
        current_time = time.time()
        should_spawn = self.should_spawn(current_time)
        if should_spawn:
            self.next_spawn_time = current_time + self.calculate_spawn_delay()
        return {'spawn_food': should_spawn}
    
    def spawn_food(self, state_snapshot):
        # Use uniform distribution for spawn position within the range
        dx = np.random.uniform(-self.spawn_range, self.spawn_range)
        dy = np.random.uniform(-self.spawn_range, self.spawn_range)
        
        # Calculate new position relative to spawner's position
        new_x = int(self.x + dx)
        new_y = int(self.y + dy)
        
        # Find an empty cell near the calculated position
        x, y = self.sim_engine.get_nearest_empty_position(new_x, new_y, state_snapshot)

        if np.random.random() < 0.1:
            params = self.high_energy_food_params
            color = (255, 0, 0)
        else:
            params = self.regular_food_params
            color = (255, 255, 0)

        energy = max(0.1, np.random.normal(params['energy'], self.entropy * params['energy']))
        nutrition = max(0.01, np.random.normal(params['nutrition'], self.entropy * params['nutrition']))

        self.sim_engine.create_item(Food, (x, y), state_snapshot, energy=energy, nutrition=nutrition, color=color)