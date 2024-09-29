import math
import random
import time
from items import Item, get_item_class
from food_spawners import FoodSpawner
from collections import deque
from state_snapshot import StateSnapshot
from typing import List, Tuple, Optional, Union, Any
from uuid import UUID


class SimulationEngine:
    def __init__(self, ui):
        self.ui = ui
        self.sim_area_widget = self.ui.sim_area
        self.sim_area_widget.sim_engine = self
        self.GRID_SIZE = 800
        self.CELL_SIZE = 10
        self.UPDATE_INTERVAL = 1.0 / ui.FPS
        self.MAX_FOOD_ITEMS = 32
        self.collision_range = 2 
        self.max_zoomorphs = 5

        self.world_width = self.GRID_SIZE
        self.world_height = self.GRID_SIZE
        self.viewport_cell_width = 0
        self.viewport_cell_height = 0
        self.viewport_cell_center_x = self.world_width // 2
        self.viewport_cell_center_y = self.world_height // 2
        
        self.state_history = deque(maxlen=100)
        self.current_state = StateSnapshot(self, current_time=time.time(), grid_size=self.GRID_SIZE)
        self.items = []
        self.organisms = []
        self.test_organism = None

        self.sim_area_widget.update_viewport_dimensions()
        self.initialize_food_spawners()

    def initialize_food_spawners(self):
        # Center of the viewport in grid coordinates
        center_x = self.viewport_cell_center_x
        center_y = self.viewport_cell_center_y

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
        
        state_snapshot.update_state_params(new_item, new_item.id)
        
        return new_item
        
    def create_organism(self, organism_class: type, position: Tuple[int, int], state_snapshot: StateSnapshot, *args, **kwargs) -> Optional[Any]:
        x, y = self.get_nearest_empty_position(position[0], position[1], state_snapshot)
        new_organism = organism_class(self, (x, y), **kwargs)
        self.organisms.append(new_organism)
        
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

    def update_simulation(self) -> None:
        new_state: StateSnapshot = self.current_state.clone_state_snapshot()
        new_state.update_time(time.time())
        self.update_all_objects(new_state)
        self.apply_simulation_state(self.current_state, new_state)
        self.state_history.append(self.current_state)
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
                    new_state.update_state_params(obj, obj_id) # in case any instance params change during apply_state

    def update_all_objects(self, new_state: StateSnapshot) -> None:
        all_objects = self.organisms + self.items
        new_state.update_snapshot_with_objects(all_objects)

    def spawn_organism(self, parent: Any, state_snapshot: StateSnapshot) -> Optional[Any]:
        parent_state = state_snapshot.get_state(parent.id)
        parent_x, parent_y = parent_state['x'], parent_state['y']
        
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        new_x = max(0, min(parent_x + dx, self.world_width - 1))
        new_y = max(0, min(parent_y + dy, self.world_height - 1))
        
        new_organism = self.create_organism(parent.__class__, (new_x, new_y), state_snapshot)
        
        return new_organism

    def get_nearest_items(self, x: int, y: int, num_items: int, detection_radius: float, state_snapshot: StateSnapshot, item_type: Optional[str] = None, return_IDs: bool = False) -> Union[List[UUID], List[Any]]:
        item_class = Item if item_type is None else get_item_class(item_type)
        
        sorted_items = sorted(
            [item for item in self.items 
             if isinstance(item, item_class) and 
             self.calculate_distance(x, y, state_snapshot.get_state(item.id)['x'], state_snapshot.get_state(item.id)['y']) <= detection_radius],
            key=lambda item: self.calculate_distance(x, y, state_snapshot.get_state(item.id)['x'], state_snapshot.get_state(item.id)['y'])
        )[:num_items]
        
        return [item.id for item in sorted_items] if return_IDs else sorted_items
    
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

    def calculate_distance(self, x1: int, y1: int, x2: int, y2: int, normalize_to_viewport: bool = False) -> float:
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if normalize_to_viewport:
            viewport_diagonal = math.sqrt(self.viewport_cell_width**2 + self.viewport_cell_height**2)
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