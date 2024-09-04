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
from typing import List, Tuple, Optional, Union, Any
from uuid import UUID

class Matrika:
    def __init__(self):
        self.GRID_SIZE = 7200
        self.CELL_SIZE = 10
        self.SCREEN_WIDTH = 1920
        self.SCREEN_HEIGHT = 1080
        self.FPS = 60
        self.UPDATE_INTERVAL = 1.0 / 30
        self.CAMERA_PAN_SPEED = 200
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
        self.current_state = StateSnapshot(self, current_time=time.time(), grid_size=self.GRID_SIZE)

        self.food_spawners = []
        self.items = []
        self.organisms = []
        self.test_organism = None
        self.visible_cells = []

        self.update_visible_cells()
        self.initialize_food_spawners()

    def initialize_food_spawners(self):
        # Center of the viewport
        center_x = self.viewport_center_x
        center_y = self.viewport_center_y

        regular_food_params = {
            'energy': 2.0,
            'nutrition': 0.25
        }

        high_energy_food_params = {
            'energy': 10.0,
            'nutrition': 2.5
        }

        spawn_range = 50
        spawn_frequency = 0.5
        entropy = 0.2

        self.create_food_spawner(center_x, center_y, regular_food_params, high_energy_food_params, 
                                   spawn_frequency=spawn_frequency, 
                                   spawn_range=spawn_range, 
                                   entropy=entropy)

    def create_food_spawner(self, x: int, y: int, regular_food_params: dict, high_energy_food_params: dict, spawn_frequency: float = 0.5, spawn_range: int = 20, entropy: float = 0.2) -> None:
        spawner = FoodSpawner(spawn_frequency, regular_food_params, high_energy_food_params, spawn_range, entropy, matrika=self)
        self.food_spawners.append(spawner)
        # Add the new spawner to the current state
        self.current_state.update_food_spawner_state(spawner.id, {
            'x': x,
            'y': y,
            'marked_for_deletion': False
        })

    def create_item(self, item_class: type, x: int, y: int, **kwargs) -> Optional[Any]:
        if 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE:
            new_item = item_class(**kwargs)
            self.items.append(new_item)
            # Add the new item to the current state
            self.current_state.update_item_state(new_item.id, {
                'x': x,
                'y': y,
                'type': type(new_item).__name__,
                'energy': new_item.energy,
                'nutrition': new_item.nutrition,
                'expiration_timer': new_item.expiration_timer,
                'reward': new_item.reward,
                'marked_for_deletion': False,
                'color': new_item.color  # Add color to the state
            })
            return new_item
        else:
            return None
        
    def create_organism(self, organism_class: type, x: int, y: int) -> Optional[Any]:
        if 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE:
            new_organism = organism_class(self, (x, y))  # Pass self (Matrika instance) and initial position
            self.organisms.append(new_organism)
            # Add the new organism to the current state
            self.current_state.update_organism_state(new_organism.id, {
                'x': x,
                'y': y,
                'marked_for_deletion': False,
                'attention_point': (x, y)  # Initialize attention point at organism's position
            })
            return new_organism 
        else:
            return None

    def is_cell_empty(self, x: int, y: int) -> bool:
        return not any(
            (item_state['x'] == x and item_state['y'] == y and not item_state.get('marked_for_deletion', False))
            for item_id, item_state in self.current_state.get_objects_in_snapshot()
            if item_state.get('type') == 'Food'
        ) and not any(
            (org_state['x'] == x and org_state['y'] == y and not org_state.get('marked_for_deletion', False))
            for org_id, org_state in self.current_state.get_objects_in_snapshot()
            if 'type' not in org_state  # Assuming organisms don't have a 'type' field
        )

    def find_empty_random_cell_near(self, x: int, y: int, max_attempts: int = 10) -> Optional[Tuple[int, int]]:
        for _ in range(max_attempts):
            dx = np.random.randint(-1, 2)
            dy = np.random.randint(-1, 2)
            new_x = max(0, min(x + dx, self.GRID_SIZE - 1))
            new_y = max(0, min(y + dy, self.GRID_SIZE - 1))
            if self.is_cell_empty(new_x, new_y):
                return new_x, new_y
        return None

    def update_viewport(self, dx: int = 0, dy: int = 0) -> None:
        self.viewport_center_x = max(self.viewport_width // 2, 
                                     min(self.viewport_center_x + dx, 
                                         self.world_width - self.viewport_width // 2))
        self.viewport_center_y = max(self.viewport_height // 2, 
                                     min(self.viewport_center_y + dy, 
                                         self.world_height - self.viewport_height // 2))
        
        self.last_viewport_update_time = time.time()
        self.update_visible_cells()

    def update_visible_cells(self) -> None:
        self.visible_cells = []
        
        viewport_left = int(self.viewport_center_x - self.viewport_width // 2)
        viewport_top = int(self.viewport_center_y - self.viewport_height // 2)
        
        for x in range(viewport_left, viewport_left + int(self.viewport_width)):
            for y in range(viewport_top, viewport_top + int(self.viewport_height)):
                if 0 <= x < self.world_width and 0 <= y < self.world_height:
                    self.visible_cells.append((x, y))

    def get_random_visible_cell(self) -> Optional[Tuple[int, int]]:
        if not self.visible_cells:
            return None
        return random.choice(self.visible_cells)

    def update_simulation(self) -> None:
        new_state: StateSnapshot = self.current_state.clone_state_snapshot()
        new_state.update_time(time.time())

        # Update all components using the new mutable state
        self.update_all_organisms(new_state)
        self.update_all_items(new_state)
        self.update_food_spawners(new_state)

        # Apply the new state
        self.apply_simulation_state(self.current_state, new_state)

        # Update the state history
        self.state_history.append(self.current_state)
        
        # Set the new state as the current state
        self.current_state = new_state

    def apply_simulation_state(self, old_state: StateSnapshot, new_state: StateSnapshot) -> None:
        for obj_id, obj_state in new_state.get_objects_in_snapshot():
            if obj_state.get('marked_for_deletion', False):
                self.remove_object(obj_id, new_state)
            else:
                obj = self.get_object_by_ID(obj_id)
                if obj:
                    new_state.apply_state_params(obj, obj_id)
                    obj.apply_state(old_state, new_state)

    def update_all_organisms(self, new_state: StateSnapshot) -> None:
        new_state.update_snapshot_with_organisms(self.organisms)

    def update_all_items(self, new_state: StateSnapshot) -> None:
        new_state.update_snapshot_with_items(self.items)

    def update_food_spawners(self, new_state: StateSnapshot) -> None:
        current_time = time.time()
        food_count = sum(1 for item in self.items if isinstance(item, Food))
        for spawner in self.food_spawners:
            spawner_state = new_state.get_state(spawner.id)
            if spawner_state and food_count < self.MAX_FOOD_ITEMS and spawner.should_spawn(current_time):
                spawn_result = spawner.spawn_food(spawner_state['x'], spawner_state['y'])
                if spawn_result:
                    new_food, spawn_x, spawn_y = spawn_result
                    food_count += 1
                    new_state.update_item_state(new_food.id, {
                        'x': spawn_x,
                        'y': spawn_y,
                        'type': type(new_food).__name__,
                        'energy': new_food.energy,
                        'nutrition': new_food.nutrition,
                        'expiration_timer': new_food.expiration_timer,
                        'reward': new_food.reward,
                        'marked_for_deletion': False,
                        'color': new_food.color
                    })

    def spawn_organism(self, parent: Any) -> Optional[Any]:
        parent_state = self.current_state.get_state(parent.id)
        parent_x, parent_y = parent_state['x'], parent_state['y']
        
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        new_x = max(0, min(parent_x + dx, self.GRID_SIZE - 1))
        new_y = max(0, min(parent_y + dy, self.GRID_SIZE - 1))
        
        new_organism = self.create_organism(parent.__class__, new_x, new_y)
        
        if new_organism:
            new_organism.clone(parent)
        
        return new_organism

    def get_nearest_items(self, x: int, y: int, num_items: int, detection_radius: float, item_type: Optional[str] = None, return_IDs: bool = False) -> Union[List[UUID], List[Any]]:
        item_class_map = {
            'food': Food
        }
        
        filtered_items = [item for item in self.items if not item_type or isinstance(item, item_class_map.get(item_type.lower(), object))]
        
        sorted_items = sorted(
            [item for item in filtered_items if self.calculate_distance(x, y, 
             self.current_state.get_state(item.id)['x'], 
             self.current_state.get_state(item.id)['y']) <= detection_radius],
            key=lambda item: self.calculate_distance(x, y, 
                             self.current_state.get_state(item.id)['x'], 
                             self.current_state.get_state(item.id)['y'])
        )
        
        if return_IDs:
            return [item.id for item in sorted_items[:num_items]]
        else:
            return sorted_items[:num_items]
    
    def item_exists(self, item_id: UUID) -> bool:
        return any(item.id == item_id for item in self.items)

    def get_item_by_ID(self, item_id: UUID) -> Optional[Any]:
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def get_organism_by_ID(self, organism_id: UUID) -> Optional[Any]:
        for organism in self.organisms:
            if organism.id == organism_id:
                return organism
        return None
    
    def get_food_spawner_by_ID(self, spawner_id: UUID) -> Optional[Any]:
        return next((spawner for spawner in self.food_spawners if spawner.id == spawner_id), None)


    def get_object_by_ID(self, obj_id: UUID) -> Optional[Any]:
        return (self.get_item_by_ID(obj_id) or 
                self.get_organism_by_ID(obj_id) or 
                self.get_food_spawner_by_ID(obj_id))
    
    def remove_object(self, obj_id: UUID, state_snapshot: StateSnapshot) -> None:
        obj = self.get_object_by_ID(obj_id)
        if obj:
            if obj in self.items:
                self.items.remove(obj)
            elif obj in self.organisms:
                self.organisms.remove(obj)
            elif obj in self.food_spawners:
                self.food_spawners.remove(obj)
        state_snapshot.remove_state(obj_id)

    def grid_to_screen(self, grid_x: int, grid_y: int) -> Tuple[int, int]:
        viewport_left = self.viewport_center_x - self.viewport_width // 2
        viewport_top = self.viewport_center_y - self.viewport_height // 2
        
        screen_x = (grid_x - viewport_left) * self.CELL_SIZE
        screen_y = (grid_y - viewport_top) * self.CELL_SIZE
        return int(screen_x), int(screen_y)

    def handle_camera_panning(self) -> None:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        dx = dy = 0

        if mouse_x <= 10:
            dx = -self.CAMERA_PAN_SPEED / self.FPS
        elif mouse_x >= self.SCREEN_WIDTH - 10:
            dx = self.CAMERA_PAN_SPEED / self.FPS

        if mouse_y <= 10:
            dy = -self.CAMERA_PAN_SPEED / self.FPS
        elif mouse_y >= self.SCREEN_HEIGHT - 10:
            dy = self.CAMERA_PAN_SPEED / self.FPS

        if dx != 0 or dy != 0:
            self.update_viewport(dx, dy)

    def calculate_organism_speed(self, organism: Any) -> float:
        return organism.movement_speed * self.simulation_fps

    def calculate_distance(self, x1: int, y1: int, x2: int, y2: int, normalize_to_viewport: bool = False) -> float:
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if normalize_to_viewport:
            viewport_diagonal = math.sqrt(self.viewport_width**2 + self.viewport_height**2)
            return distance / viewport_diagonal
        return distance
    
    def calculate_angle(self, x1: int, y1: int, x2: int, y2: int, normalize: bool = False) -> float:
        dx = x2 - x1
        dy = y2 - y1
        angle = math.atan2(dy, dx)
        
        if normalize:
            # Normalize angle to [0, 1] range
            normalized_angle = (angle + math.pi) / (2 * math.pi)
            return normalized_angle
        
        return angle