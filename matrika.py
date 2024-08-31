import numpy as np
import math
import random
import pygame
import time
import uuid
from items import Food
from food_spawners import FoodSpawner
from collections import deque
from immutable_state_view import ImmutableStateView

class Matrika:
    def __init__(self):
        self.GRID_SIZE = 7200
        self.CELL_SIZE = 10
        self.SCREEN_WIDTH = 1920
        self.SCREEN_HEIGHT = 1080
        self.FPS = 60
        self.UPDATE_INTERVAL = 1.0 / 30
        self.CAMERA_PAN_SPEED = 800
        self.MAX_FOOD_ITEMS = 12

        # Colors
        self.BLACK = (0, 0, 0)
        self.NEON_GREEN = (57, 255, 20)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)

        self.viewport_width = self.SCREEN_WIDTH // self.CELL_SIZE
        self.viewport_height = self.SCREEN_HEIGHT // self.CELL_SIZE
        
        # Center the viewport
        self.viewport_x = (self.GRID_SIZE - self.viewport_width) // 2
        self.viewport_y = (self.GRID_SIZE - self.viewport_height) // 2
        
        self.prev_viewport_x = self.viewport_x
        self.prev_viewport_y = self.viewport_y
        self.last_viewport_update_time = time.time()
        self.last_pan_time = time.time()
        
        self.state_history = deque(maxlen=100)  # Adjust maxlen as needed
        self.current_state = self.create_state_snapshot()

        self.food_spawners = []
        self.items = []
        self.organisms = []
        self.test_organism = None
        self.visible_cells = []

        # Now initialize food spawners after centering the viewport
        self.initialize_food_spawners()

    def initialize_food_spawners(self):
        margin = 15  # Cells inward from the corners
        spawner_positions = [
            (self.viewport_x + margin, self.viewport_y + margin),
            (self.viewport_x + self.viewport_width - margin, self.viewport_y + margin),
            (self.viewport_x + margin, self.viewport_y + self.viewport_height - margin),
            (self.viewport_x + self.viewport_width - margin, self.viewport_y + self.viewport_height - margin)
        ]

        regular_food_params = {
            'energy': 2.0,
            'nutrition': 0.25
        }

        high_energy_food_params = {
            'energy': 20.0,
            'nutrition': 2.5
        }

        spawn_range = 20
        spawn_frequency = 0.5
        entropy = 0.2

        for x, y in spawner_positions:
            self.create_food_spawner(x, y, regular_food_params, high_energy_food_params, 
                                       spawn_frequency=spawn_frequency, 
                                       spawn_range=spawn_range, 
                                       entropy=entropy)

    def create_food_spawner(self, x, y, regular_food_params, high_energy_food_params, spawn_frequency=0.5, spawn_range=20, entropy=0.2):
        spawner = FoodSpawner(spawn_frequency, regular_food_params, high_energy_food_params, spawn_range, entropy, matrika=self)
        self.food_spawners.append(spawner)
        # Add the new spawner to the current state
        self.current_state['food_spawners'][str(spawner.id)] = {
            'x': x,
            'y': y,
            'marked_for_deletion': False
        }

    def create_item(self, item_class, x, y, **kwargs):
        if 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE:
            new_item = item_class(**kwargs)
            self.items.append(new_item)
            # Add the new item to the current state
            self.current_state['items'][str(new_item.id)] = {
                'x': x,
                'y': y,
                'type': type(new_item).__name__,
                'energy': new_item.energy,
                'nutrition': new_item.nutrition,
                'expiration_timer': new_item.expiration_timer,
                'reward': new_item.reward,
                'marked_for_deletion': False,
                'color': new_item.color  # Add color to the state
            }
            return new_item
        else:
            return None
        
    def create_organism(self, organism_class, x, y):
        if 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE:
            new_organism = organism_class(self, (x, y))  # Pass self (Matrika instance) and initial position
            self.organisms.append(new_organism)
            # Add the new organism to the current state
            self.current_state['organisms'][str(new_organism.id)] = {
                'x': x,
                'y': y,
                'marked_for_deletion': False,
                'attention_point': (x, y)  # Initialize attention point at organism's position
            }
            return new_organism 
        else:
            return None

    def is_cell_empty(self, x, y):
        return not any(
            (item_state['x'] == x and item_state['y'] == y and not item_state.get('marked_for_deletion', False))
            for item_state in self.current_state['items'].values()
        ) and not any(
            (org_state['x'] == x and org_state['y'] == y and not org_state.get('marked_for_deletion', False))
            for org_state in self.current_state['organisms'].values()
        )

    def find_empty_random_cell_near(self, x, y, max_attempts=10):
        for _ in range(max_attempts):
            dx = np.random.randint(-1, 2)
            dy = np.random.randint(-1, 2)
            new_x = max(0, min(x + dx, self.GRID_SIZE - 1))
            new_y = max(0, min(y + dy, self.GRID_SIZE - 1))
            if self.is_cell_empty(new_x, new_y):
                return new_x, new_y
        return None

    def update_viewport(self, dx=0, dy=0):
        self.prev_viewport_x = self.viewport_x
        self.prev_viewport_y = self.viewport_y
        
        self.viewport_x += dx
        self.viewport_y += dy
        
        self.last_viewport_update_time = time.time()
        self.update_visible_cells()

    def update_visible_cells(self):
        self.visible_cells = []
        for x in range(int(self.viewport_x), int(self.viewport_x + self.viewport_width) + 1):
            for y in range(int(self.viewport_y), int(self.viewport_y + self.viewport_height) + 1):
                if 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE:
                    self.visible_cells.append((x, y))

    def get_random_visible_cell(self):
        if not self.visible_cells:
            return None
        return random.choice(self.visible_cells)

    def update_all_organisms(self, state):
        for org_id, org_state in list(state['organisms'].items()):
            if org_state.get('marked_for_deletion', False):
                continue
            
            organism = next(org for org in self.organisms if str(org.id) == org_id)
            
            immutable_state = ImmutableStateView(state)
            state_change_dict = organism.update_state(immutable_state)
            
            self.process_external_state_change(state, org_id, org_state, state_change_dict, organism)
        
        return state

    def process_external_state_change(self, state, org_id, org_state, state_change_dict, organism):
        if 'movement_vector' in state_change_dict:
            new_x, new_y = self.calculate_new_position(
                org_state['x'], 
                org_state['y'], 
                state_change_dict['movement_vector']
            )
            
            collision_object = self.handle_collision(new_x, new_y, state, organism)
            if collision_object is None:
                state['organisms'][org_id]['x'] = new_x
                state['organisms'][org_id]['y'] = new_y
            elif isinstance(collision_object, Food):
                collision_object.consume(organism)
                state['items'][str(collision_object.id)]['marked_for_deletion'] = True
                state['organisms'][org_id]['x'] = new_x
                state['organisms'][org_id]['y'] = new_y
        
        if 'attention_point' in state_change_dict:
            state['organisms'][org_id]['attention_point'] = state_change_dict['attention_point']
        
        if 'alive' in state_change_dict and not state_change_dict['alive']:
            state['organisms'][org_id]['marked_for_deletion'] = True
        
        if state_change_dict.get('spawn', False):
            self.spawn_organism(organism)

    def handle_collision(self, x, y, state, organism):
        for item_id, item_state in state['items'].items():
            if item_state['x'] == x and item_state['y'] == y and not item_state.get('marked_for_deletion', False):
                item = self.get_item_by_ID(uuid.UUID(item_id))
                if item:
                    return item
        
        for org_id, org_state in state['organisms'].items():
            if org_id != str(organism.id) and org_state['x'] == x and org_state['y'] == y and not org_state.get('marked_for_deletion', False):
                return self.get_organism_by_ID(uuid.UUID(org_id))
        
        return None
        
    def calculate_new_position(self, x, y, movement_vector):
        dx, dy = movement_vector
        new_x = max(0, min(x + dx, self.GRID_SIZE - 1))
        new_y = max(0, min(y + dy, self.GRID_SIZE - 1))
        return new_x, new_y

    def is_position_empty(self, x, y, state):
        for org_state in state['organisms'].values():
            if org_state['x'] == x and org_state['y'] == y:
                return False
        for item_state in state['items'].values():
            if item_state['x'] == x and item_state['y'] == y:
                return False
        return True

    def update_expiration_timers(self, state):
        for item_id, item_state in state['items'].items():
            if item_state.get('marked_for_deletion', False):
                continue
            if item_state['expiration_timer'] > 0:
                item_state['expiration_timer'] -= self.UPDATE_INTERVAL
                if item_state['expiration_timer'] <= 0:
                    item_state['marked_for_deletion'] = True
        return state

    def update_food_spawners(self, state):
        current_time = time.time()
        food_count = sum(1 for item in self.items if isinstance(item, Food))
        for spawner in self.food_spawners:
            spawner_state = state['food_spawners'][str(spawner.id)]
            if food_count < self.MAX_FOOD_ITEMS and spawner.should_spawn(current_time):
                new_food = spawner.spawn_food(spawner_state['x'], spawner_state['y'])
                if new_food:
                    food_count += 1
        return state

    def apply_simulation_state(self):
        old_state = self.state_history[-1] if self.state_history else None
        new_state = self.current_state

        # Update organisms
        for organism in list(self.organisms):
            org_id = str(organism.id)
            new_org_state = new_state['organisms'].get(org_id)
            if new_org_state:
                if new_org_state.get('marked_for_deletion', False):
                    self.remove_organism(organism)
                elif old_state:
                    organism.apply_state(old_state, new_state)

        # Update items
        for item in list(self.items):
            item_id = str(item.id)
            new_item_state = new_state['items'].get(item_id)
            if new_item_state:
                if new_item_state.get('marked_for_deletion', False):
                    self.remove_item(item)
                elif old_state:
                    item.apply_state(old_state, new_state)

    def get_interpolated_viewport(self):
        current_time = time.time()
        time_since_update = current_time - self.last_viewport_update_time
        interpolation_factor = min(time_since_update / self.UPDATE_INTERVAL, 1.0)
        
        interp_x = self.prev_viewport_x + (self.viewport_x - self.prev_viewport_x) * interpolation_factor
        interp_y = self.prev_viewport_y + (self.viewport_y - self.prev_viewport_y) * interpolation_factor
        
        return interp_x, interp_y

    def grid_to_screen(self, grid_x, grid_y):
        interp_viewport_x, interp_viewport_y = self.get_interpolated_viewport()
        screen_x = (grid_x - interp_viewport_x) * self.CELL_SIZE
        screen_y = (grid_y - interp_viewport_y) * self.CELL_SIZE
        return int(screen_x), int(screen_y)

    def handle_camera_panning(self):
        current_time = time.time()
        elapsed_time = current_time - self.last_pan_time
        
        if elapsed_time >= 1 / self.FPS:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            base_pan_speed = self.CAMERA_PAN_SPEED
            
            dx = dy = 0.0

            if mouse_x <= 10:
                dx = -1
            elif mouse_x >= self.SCREEN_WIDTH - 10:
                dx = 1

            if mouse_y <= 10:
                dy = -1
            elif mouse_y >= self.SCREEN_HEIGHT - 10:
                dy = 1

            current_direction = (dx, dy)
            if not hasattr(self, 'last_pan_direction') or self.last_pan_direction != current_direction:
                self.continuous_pan_start = current_time
                pan_speed = base_pan_speed
            elif current_time - self.continuous_pan_start >= 0.1:
                pan_speed = base_pan_speed * 2.0
            else:
                pan_speed = base_pan_speed

            self.last_pan_direction = current_direction

            pan_distance = pan_speed * elapsed_time
            dx *= pan_distance
            dy *= pan_distance

            if dx != 0 or dy != 0:
                new_viewport_x = self.viewport_x + dx / self.CELL_SIZE
                new_viewport_y = self.viewport_y + dy / self.CELL_SIZE

                new_viewport_x = max(0, min(new_viewport_x, self.GRID_SIZE - self.viewport_width))
                new_viewport_y = max(0, min(new_viewport_y, self.GRID_SIZE - self.viewport_height))

                actual_dx = new_viewport_x - self.viewport_x
                actual_dy = new_viewport_y - self.viewport_y

                self.update_viewport(actual_dx, actual_dy)
            else:
                self.continuous_pan_start = current_time

            self.last_pan_time = current_time

    def remove_item(self, item):
        if item in self.items:
            self.items.remove(item)
        item_id = str(item.id)
        if item_id in self.current_state['items']:
            del self.current_state['items'][item_id]

    def remove_organism(self, organism):
        if organism in self.organisms:
            self.organisms.remove(organism)
        org_id = str(organism.id)
        if org_id in self.current_state['organisms']:
            del self.current_state['organisms'][org_id]

    def calculate_organism_speed(self, organism):
        return organism.movement_speed * self.simulation_fps

    def calculate_distance(self, x1, y1, x2, y2, normalize_to_viewport=False):
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if normalize_to_viewport:
            viewport_diagonal = math.sqrt(self.viewport_width**2 + self.viewport_height**2)
            return distance / viewport_diagonal
        return distance

    def update_simulation(self):
        new_state = self.current_state.copy()
        
        new_state['time'] = time.time()  # Update the time at the beginning of the simulation update
        
        new_state = self.update_all_organisms(new_state)
        new_state = self.update_expiration_timers(new_state)
        new_state = self.update_food_spawners(new_state)
        
        self.current_state = new_state
        self.apply_simulation_state()
        self.state_history.append(self.current_state)
        
       

    def spawn_organism(self, parent):
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        new_x = max(0, min(parent.x + dx, self.GRID_SIZE - 1))
        new_y = max(0, min(parent.y + dy, self.GRID_SIZE - 1))
        
        new_organism = self.create_organism(parent.__class__, new_x, new_y)
        
        if new_organism:
            new_organism.clone(parent)
        
        return new_organism

    def get_nearest_items(self, x, y, num_items, item_type=None):
        item_class_map = {
            'food': Food
        }
        
        if item_type and item_type.lower() in item_class_map:
            filtered_items = [item for item in self.items if isinstance(item, item_class_map[item_type.lower()])]
        else:
            filtered_items = self.items
        
        sorted_items = sorted(filtered_items, key=lambda item: self.calculate_distance(x, y, 
                              self.current_state['items'][str(item.id)]['x'], 
                              self.current_state['items'][str(item.id)]['y']))
        return sorted_items[:num_items]

    def calculate_angle(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return math.atan2(dy, dx)
    
    def item_exists(self, item_id):
        return any(item.id == item_id for item in self.items)

    def get_item_by_ID(self, item_id: uuid.UUID):
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def get_organism_by_ID(self, organism_id: uuid.UUID):
        for organism in self.organisms:
            if organism.id == organism_id:
                return organism
        return None

    def create_state_snapshot(self): 
        state = {
            'organisms': {},
            'items': {},
            'food_spawners': {},
            'grid_size': self.GRID_SIZE,
            'time': time.time()
        }

        return state