import numpy as np
import time
from items import Food
import uuid

class FoodSpawner:
    def __init__(self, spawn_frequency, regular_food_params, high_energy_food_params, spawn_range, entropy, matrika):
        self.id = uuid.uuid4()
        self.spawn_frequency = spawn_frequency
        self.regular_food_params = regular_food_params
        self.high_energy_food_params = high_energy_food_params
        self.spawn_range = spawn_range
        self.entropy = entropy
        self.matrika = matrika
        self.next_spawn_time = time.time()

    def calculate_spawn_delay(self):
        base_delay = 1 / self.spawn_frequency
        noise = np.random.normal(0, self.entropy * base_delay)
        return max(0.1, base_delay + noise)

    def should_spawn(self, current_time):
        return current_time >= self.next_spawn_time

    def spawn_food(self, x, y):
        # Use uniform distribution for spawn position within the range
        dx = np.random.uniform(-self.spawn_range, self.spawn_range)
        dy = np.random.uniform(-self.spawn_range, self.spawn_range)
        
        # Calculate new position relative to spawner's position
        new_x = int(x + dx)
        new_y = int(y + dy)
        
        # Ensure the new position is within the grid bounds
        new_x = max(0, min(new_x, self.matrika.GRID_SIZE - 1))
        new_y = max(0, min(new_y, self.matrika.GRID_SIZE - 1))

        # Find an empty cell near the calculated position
        empty_cell = self.matrika.find_empty_random_cell_near(new_x, new_y)
        if empty_cell is None:
            self.next_spawn_time = time.time() + self.calculate_spawn_delay()
            return None

        x, y = empty_cell

        if np.random.random() < 0.1:
            params = self.high_energy_food_params
            color = (255, 0, 0)
        else:
            params = self.regular_food_params
            color = (255, 255, 0)

        energy = max(0.1, np.random.normal(params['energy'], self.entropy * params['energy']))
        nutrition = max(0.01, np.random.normal(params['nutrition'], self.entropy * params['nutrition']))

        new_food = self.matrika.create_item(Food, x, y, energy=energy, nutrition=nutrition, color=color)
        self.next_spawn_time = time.time() + self.calculate_spawn_delay()
        return new_food