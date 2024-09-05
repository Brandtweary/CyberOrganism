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

        self.items = []
        self.organisms = []
        self.test_organism = None
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

        self.create_item(
            FoodSpawner,
            (center_x, center_y),
            self.current_state,
            spawn_frequency=spawn_frequency,
            regular_food_params=regular_food_params,
            high_energy_food_params=high_energy_food_params,
            spawn_range=spawn_range,
            entropy=entropy
        )

    def create_item(self, item_class: type, position: Tuple[int, int], state_snapshot: StateSnapshot, *args, **kwargs) -> Optional[Any]:
        x, y = self.get_nearest_empty_position(position[0], position[1], state_snapshot)
        new_item = item_class(self, (x, y), **kwargs)
        self.items.append(new_item)
        
        state_snapshot.add_state(new_item.id, {})
        state_snapshot.update_state_params(new_item, new_item.id)
        
        return new_item
        
    def create_organism(self, organism_class: type, position: Tuple[int, int], state_snapshot: StateSnapshot, *args, **kwargs) -> Optional[Any]:
        x, y = self.get_nearest_empty_position(position[0], position[1], state_snapshot)
        new_organism = organism_class(self, (x, y), **kwargs)
        self.organisms.append(new_organism)
        
        state_snapshot.add_state(new_organism.id, {})
        state_snapshot.update_state_params(new_organism, new_organism.id)
        
        return new_organism

    def is_cell_empty(self, x: int, y: int, state_snapshot: StateSnapshot) -> bool:
        return not any(
            (obj_state['x'] == x and obj_state['y'] == y and 
             not obj_state.get('marked_for_deletion', False) and
             obj_state.get('collision', True))
            for _, obj_state in state_snapshot.get_objects_in_snapshot()
        )

    def get_nearest_empty_position(self, x: int, y: int, state_snapshot: StateSnapshot) -> Tuple[int, int]:
        # Ensure the position is within the grid
        x = max(0, min(x, self.world_width - 1))
        y = max(0, min(y, self.world_height - 1))
        
        if self.is_cell_empty(x, y, state_snapshot):
            return x, y
        
        # If the cell is occupied, find the nearest empty cell
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            distance = 1
            while True:
                new_x = x + dx * distance
                new_y = y + dy * distance
                
                if not (0 <= new_x < self.world_width and 0 <= new_y < self.world_height):
                    break  # Out of bounds, try next direction
                
                if self.is_cell_empty(new_x, new_y, state_snapshot):
                    return new_x, new_y
                
                distance += 1
        
        # If no empty cell found, return None or raise an exception
        raise ValueError("No empty cell found in the entire grid")

    def update_viewport(self, dx: int = 0, dy: int = 0) -> None:
        self.viewport_center_x = max(self.viewport_width // 2, 
                                     min(self.viewport_center_x + dx, 
                                         self.world_width - self.viewport_width // 2))
        self.viewport_center_y = max(self.viewport_height // 2, 
                                     min(self.viewport_center_y + dy, 
                                         self.world_height - self.viewport_height // 2))
        
        self.last_viewport_update_time = time.time()

    def update_simulation(self) -> None:
        new_state: StateSnapshot = self.current_state.clone_state_snapshot()
        new_state.update_time(time.time())

        # Update all components using the new mutable state
        self.update_all_organisms(new_state)
        self.update_all_items(new_state)

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

    def spawn_organism(self, parent: Any, state_snapshot: StateSnapshot) -> Optional[Any]:
        parent_state = state_snapshot.get_state(parent.id)
        parent_x, parent_y = parent_state['x'], parent_state['y']
        
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        new_x = max(0, min(parent_x + dx, self.world_width - 1))
        new_y = max(0, min(parent_y + dy, self.world_height - 1))
        
        new_organism = self.create_organism(parent.__class__, (new_x, new_y), state_snapshot)
        
        if new_organism:
            new_organism.clone(parent)
        
        return new_organism

    def get_nearest_items(self, x: int, y: int, num_items: int, detection_radius: float, state_snapshot: StateSnapshot, item_type: Optional[str] = None, return_IDs: bool = False) -> Union[List[UUID], List[Any]]:
        item_class_map = {
            'food': Food
        }
        
        filtered_items = [item for item in self.items if not item_type or isinstance(item, item_class_map.get(item_type.lower(), object))]
        
        sorted_items = sorted(
            [item for item in filtered_items if self.calculate_distance(x, y, 
             state_snapshot.get_state(item.id)['x'], 
             state_snapshot.get_state(item.id)['y']) <= detection_radius],
            key=lambda item: self.calculate_distance(x, y, 
                             state_snapshot.get_state(item.id)['x'], 
                             state_snapshot.get_state(item.id)['y'])
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

    def get_object_by_ID(self, obj_id: UUID) -> Optional[Any]:
        return (self.get_item_by_ID(obj_id) or 
                self.get_organism_by_ID(obj_id))
    
    def remove_object(self, obj_id: UUID, state_snapshot: StateSnapshot) -> None:
        obj = self.get_object_by_ID(obj_id)
        if obj:
            if obj in self.items:
                self.items.remove(obj)
            elif obj in self.organisms:
                self.organisms.remove(obj)
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