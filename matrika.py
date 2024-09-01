import numpy as np
import math
import random
import pygame
import time
import uuid
from items import Food
from food_spawners import FoodSpawner
from collections import deque
from state_snapshot import StateSnapshot
import copy

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
        self.collision_range = 2  # Add collision range parameter

        # Colors
        self.BLACK = (0, 0, 0)
        self.NEON_GREEN = (57, 255, 20)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)

        self.world_width = self.GRID_SIZE
        self.world_height = self.GRID_SIZE
        
        self.viewport_width = self.SCREEN_WIDTH // self.CELL_SIZE
        self.viewport_height = self.SCREEN_HEIGHT // self.CELL_SIZE
        
        # Center of the viewport in world coordinates
        self.viewport_center_x = self.world_width // 2
        self.viewport_center_y = self.world_height // 2
        
        self.last_viewport_update_time = time.time()
        self.last_pan_time = time.time()
        
        self.state_history = deque(maxlen=100)  # Adjust maxlen as needed
        self.current_state = StateSnapshot(self.create_state_snapshot(), mutable=True)

        self.food_spawners = []
        self.items = []
        self.organisms = []
        self.test_organism = None
        self.visible_cells = []

        self.update_visible_cells()
        self.initialize_food_spawners()

    def initialize_food_spawners(self):
        margin = 15  # Cells inward from the corners
        spawner_positions = [
            (self.viewport_center_x - self.viewport_width // 2 + margin, self.viewport_center_y - self.viewport_height // 2 + margin),
            (self.viewport_center_x + self.viewport_width // 2 - margin, self.viewport_center_y - self.viewport_height // 2 + margin),
            (self.viewport_center_x - self.viewport_width // 2 + margin, self.viewport_center_y + self.viewport_height // 2 - margin),
            (self.viewport_center_x + self.viewport_width // 2 - margin, self.viewport_center_y + self.viewport_height // 2 - margin)
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
        self.viewport_center_x = max(self.viewport_width // 2, 
                                     min(self.viewport_center_x + dx, 
                                         self.world_width - self.viewport_width // 2))
        self.viewport_center_y = max(self.viewport_height // 2, 
                                     min(self.viewport_center_y + dy, 
                                         self.world_height - self.viewport_height // 2))
        
        self.last_viewport_update_time = time.time()
        self.update_visible_cells()

    def update_visible_cells(self):
        self.visible_cells = []
        
        viewport_left = int(self.viewport_center_x - self.viewport_width // 2)
        viewport_top = int(self.viewport_center_y - self.viewport_height // 2)
        
        for x in range(viewport_left, viewport_left + int(self.viewport_width)):
            for y in range(viewport_top, viewport_top + int(self.viewport_height)):
                if 0 <= x < self.world_width and 0 <= y < self.world_height:
                    self.visible_cells.append((x, y))

    def get_random_visible_cell(self):
        if not self.visible_cells:
            return None
        return random.choice(self.visible_cells)

    def update_simulation(self):
        # Create a new state as a deep copy of the current state
        new_state = StateSnapshot(copy.deepcopy(self.current_state._state), mutable=True)

        new_state['time'] = time.time()

        # Update all components using the new mutable state
        self.update_all_organisms(new_state)
        self.update_expiration_timers(new_state)
        self.update_food_spawners(new_state)

        # Apply the new state
        self.apply_simulation_state(self.current_state, new_state)

        # Update the state history
        self.state_history.append(self.current_state)
        
        # Set the new state as the current state
        self.current_state = new_state

    def apply_simulation_state(self, old_state, new_state):
        # Update items
        for item_id, item_state in list(new_state['items'].items()):
            if item_state.get('marked_for_deletion', False):
                del new_state['items'][item_id]
                item = next((item for item in self.items if str(item.id) == item_id), None)
                if item:
                    self.items.remove(item)
            else:
                item = next((item for item in self.items if str(item.id) == item_id), None)
                if item:
                    item.apply_state(old_state, new_state)

        # Update organisms
        for org_id, org_state in list(new_state['organisms'].items()):
            if org_state.get('marked_for_deletion', False):
                del new_state['organisms'][org_id]
                organism = next((org for org in self.organisms if str(org.id) == org_id), None)
                if organism:
                    self.organisms.remove(organism)
            else:
                organism = next((org for org in self.organisms if str(org.id) == org_id), None)
                if organism:
                    organism.apply_state(old_state, new_state)

    def update_all_organisms(self, new_state):
        for org_id, org_state in list(new_state['organisms'].items()):
            if org_state.get('marked_for_deletion', False):
                continue
            
            organism = next(org for org in self.organisms if str(org.id) == org_id)
            
            state_change_dict = organism.update_state(new_state)
            
            self.process_external_state_change(org_id, org_state, state_change_dict, organism, new_state)

    def update_expiration_timers(self, new_state):
        for item_id, item_state in new_state['items'].items():
            if item_state.get('marked_for_deletion', False):
                continue
            if item_state['expiration_timer'] > 0:
                item_state['expiration_timer'] -= self.UPDATE_INTERVAL
                if item_state['expiration_timer'] <= 0:
                    item_state['marked_for_deletion'] = True

    def update_food_spawners(self, new_state):
        current_time = time.time()
        food_count = sum(1 for item in self.items if isinstance(item, Food))
        for spawner in self.food_spawners:
            spawner_state = new_state['food_spawners'][str(spawner.id)]
            if food_count < self.MAX_FOOD_ITEMS and spawner.should_spawn(current_time):
                spawn_result = spawner.spawn_food(spawner_state['x'], spawner_state['y'])
                if spawn_result:
                    new_food, spawn_x, spawn_y = spawn_result
                    food_count += 1
                    # Add the new food to the new state
                    new_state['items'][str(new_food.id)] = {
                        'x': spawn_x,
                        'y': spawn_y,
                        'type': type(new_food).__name__,
                        'energy': new_food.energy,
                        'nutrition': new_food.nutrition,
                        'expiration_timer': new_food.expiration_timer,
                        'reward': new_food.reward,
                        'marked_for_deletion': False,
                        'color': new_food.color
                    }

    def process_external_state_change(self, org_id, org_state, state_change_dict, organism, new_state):
        if 'movement_vector' in state_change_dict:
            new_x, new_y = self.calculate_new_position(
                org_state['x'], 
                org_state['y'], 
                state_change_dict['movement_vector']
            )
            
            collision_objects = self.handle_collision(new_x, new_y, organism, new_state)
            if not collision_objects:
                new_state['organisms'][org_id]['x'] = new_x
                new_state['organisms'][org_id]['y'] = new_y
            else:
                for obj in collision_objects:
                    if isinstance(obj, Food):
                        obj.consume(organism)
                        new_state['items'][str(obj.id)]['marked_for_deletion'] = True
        
        if 'attention_vector' in state_change_dict:
            current_org_x, current_org_y = org_state['x'], org_state['y']
            current_attention_x, current_attention_y = org_state.get('attention_point', (current_org_x, current_org_y))
            dx, dy = state_change_dict['attention_vector']
            new_attention_x = current_attention_x + dx
            new_attention_y = current_attention_y + dy
            
            # Use the organism's detection radius directly
            detection_radius = organism.detection_radius
            
            # Calculate the distance between the new attention point and the organism
            distance = self.calculate_distance(current_org_x, current_org_y, new_attention_x, new_attention_y)
            
            if distance <= detection_radius:
                new_state['organisms'][org_id]['attention_point'] = (new_attention_x, new_attention_y)
            else:
                # If the new attention point is outside the detection radius,
                # we constrain it to the edge of the detection radius
                angle = self.calculate_angle(current_org_x, current_org_y, new_attention_x, new_attention_y)
                constrained_x = current_org_x + detection_radius * math.cos(angle)
                constrained_y = current_org_y + detection_radius * math.sin(angle)
                new_state['organisms'][org_id]['attention_point'] = (constrained_x, constrained_y)
        
        if 'nearest_item_id' in state_change_dict:
            new_state['organisms'][org_id]['nearest_item_id'] = state_change_dict['nearest_item_id']
        
        if 'alive' in state_change_dict and not state_change_dict['alive']:
            new_state['organisms'][org_id]['marked_for_deletion'] = True
        
        if state_change_dict.get('spawn', False):
            self.spawn_organism(organism)

    def is_point_visible(self, x, y):
        viewport_left = self.viewport_center_x - self.viewport_width // 2
        viewport_top = self.viewport_center_y - self.viewport_height // 2
        viewport_right = viewport_left + self.viewport_width
        viewport_bottom = viewport_top + self.viewport_height

        return viewport_left <= x < viewport_right and viewport_top <= y < viewport_bottom

    def handle_collision(self, x, y, organism, new_state):
        collision_objects = []
        for item_id, item_state in new_state['items'].items():
            if self.calculate_distance(x, y, item_state['x'], item_state['y']) <= self.collision_range and not item_state.get('marked_for_deletion', False):
                item = self.get_item_by_ID(item_id)
                if item:
                    collision_objects.append(item)
        
        for org_id, org_state in new_state['organisms'].items():
            if org_id != str(organism.id) and self.calculate_distance(x, y, org_state['x'], org_state['y']) <= self.collision_range and not org_state.get('marked_for_deletion', False):
                collision_objects.append(self.get_organism_by_ID(org_id))
        
        return collision_objects
        
    def calculate_new_position(self, x, y, movement_vector):
        dx, dy = movement_vector
        new_x = max(0, min(x + dx, self.GRID_SIZE - 1))
        new_y = max(0, min(y + dy, self.GRID_SIZE - 1))
        return new_x, new_y

    def is_position_empty(self, x, y):
        for org_state in self.current_state['organisms'].values():
            if org_state['x'] == x and org_state['y'] == y:
                return False
        for item_state in self.current_state['items'].values():
            if item_state['x'] == x and item_state['y'] == y:
                return False
        return True

    def spawn_organism(self, parent):
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        new_x = max(0, min(parent.x + dx, self.GRID_SIZE - 1))
        new_y = max(0, min(parent.y + dy, self.GRID_SIZE - 1))
        
        new_organism = self.create_organism(parent.__class__, new_x, new_y)
        
        if new_organism:
            new_organism.clone(parent)
        
        return new_organism

    def get_nearest_items(self, x, y, num_items, detection_radius, item_type=None, return_IDs=False):
        item_class_map = {
            'food': Food
        }
        
        if item_type and item_type.lower() in item_class_map:
            filtered_items = [item for item in self.items if isinstance(item, item_class_map[item_type.lower()])]
        else:
            filtered_items = self.items
        
        sorted_items = sorted(
            [item for item in filtered_items if self.calculate_distance(x, y, 
             self.current_state['items'][str(item.id)]['x'], 
             self.current_state['items'][str(item.id)]['y']) <= detection_radius],
            key=lambda item: self.calculate_distance(x, y, 
                             self.current_state['items'][str(item.id)]['x'], 
                             self.current_state['items'][str(item.id)]['y'])
        )
        
        if return_IDs:
            return [str(item.id) for item in sorted_items[:num_items]]
        else:
            return sorted_items[:num_items]

    def calculate_angle(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return math.atan2(dy, dx)
    
    def item_exists(self, item_id):
        return any(item.id == item_id for item in self.items)

    def get_item_by_ID(self, item_id):
        if isinstance(item_id, str):
            item_id = uuid.UUID(item_id)
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def get_organism_by_ID(self, organism_id):
        if isinstance(organism_id, str):
            organism_id = uuid.UUID(organism_id)
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

    def grid_to_screen(self, grid_x, grid_y):
        viewport_left = self.viewport_center_x - self.viewport_width // 2
        viewport_top = self.viewport_center_y - self.viewport_height // 2
        
        screen_x = (grid_x - viewport_left) * self.CELL_SIZE
        screen_y = (grid_y - viewport_top) * self.CELL_SIZE
        return int(screen_x), int(screen_y)

    def handle_camera_panning(self):
        current_time = time.time()
        elapsed_time = current_time - self.last_pan_time
        
        if elapsed_time >= 1 / self.FPS:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            pan_speed = self.CAMERA_PAN_SPEED * elapsed_time
            
            dx = dy = 0

            if mouse_x <= 10:
                dx = -pan_speed
            elif mouse_x >= self.SCREEN_WIDTH - 10:
                dx = pan_speed

            if mouse_y <= 10:
                dy = -pan_speed
            elif mouse_y >= self.SCREEN_HEIGHT - 10:
                dy = pan_speed

            if dx != 0 or dy != 0:
                self.update_viewport(dx, dy)

            self.last_pan_time = current_time

    def calculate_organism_speed(self, organism):
        return organism.movement_speed * self.simulation_fps

    def calculate_distance(self, x1, y1, x2, y2, normalize_to_viewport=False):
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if normalize_to_viewport:
            viewport_diagonal = math.sqrt(self.viewport_width**2 + self.viewport_height**2)
            return distance / viewport_diagonal
        return distance