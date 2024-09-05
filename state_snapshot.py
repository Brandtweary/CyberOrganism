from typing import Dict, Any, Optional, List, Tuple, Union
from uuid import UUID
import math
import copy
from enums import ObjectType


class StateSnapshot:
    def __init__(self, SimulationEngine, current_time: float, grid_size: int):
        self._state = {
            'object_states': {},
            'grid_size': grid_size,
            'time': current_time
        }
        self.sim_engine = SimulationEngine

    def clone_state_snapshot(self) -> 'StateSnapshot':
        return copy.deepcopy(self)

    def get_state(self, uuid: UUID) -> Optional[Dict[str, Any]]:
        return self._state['object_states'].get(uuid)
    
    def get_objects_in_snapshot(self, filter_type: Optional[ObjectType] = None) -> List[Tuple[UUID, Dict[str, Any]]]:
        filtered_objects = []
        for uuid, state in self._state['object_states'].items():
            obj = self.sim_engine.get_object_by_ID(uuid)
            if obj:
                if filter_type is None or obj.type == filter_type:
                    filtered_objects.append((uuid, state))
            else:
                # Object should have been deleted, mark it for deletion
                state['marked_for_deletion'] = True
        
        return filtered_objects
    
    def add_state(self, uuid: UUID, state: Dict[str, Any]):
        self._state['object_states'][uuid] = state

    def update_state(self, uuid: UUID, state: Dict[str, Any]):
        self._state['object_states'][uuid] = state

    def remove_state(self, uuid: UUID):
        self._state['object_states'].pop(uuid, None)

    def update_time(self, new_time: float):
        self._state['time'] = new_time

    def update_grid_size(self, new_grid_size: int):
        self._state['grid_size'] = new_grid_size

    def calculate_synchronized_params(self, instance: Any, state: Dict[str, Any]) -> List[str]:
        if not hasattr(instance, 'synchronized_params') or not hasattr(instance, 'param_count') or len(state) != instance.param_count:
            synchronized_params = [
                param for param, value in instance.__dict__.items()
                if isinstance(value, (int, float, str, bool, tuple, UUID)) or 
                (param.endswith('_ID') and value is None)
            ]
            instance.synchronized_params = synchronized_params
            instance.param_count = len(synchronized_params)
        return instance.synchronized_params

    def update_state_params(self, instance: Any, uuid: UUID):
        state = self.get_state(uuid)
        if state is None:
            state = {}
            self.add_state(uuid, state)
        
        synchronized_params = self.calculate_synchronized_params(instance, state)
        
        for param in synchronized_params:
            value = getattr(instance, param)
            if param not in state or state[param] != value:
                state[param] = value
        
        self.update_state(uuid, state)

    def apply_state_params(self, instance: Any, uuid: UUID):
        state = self.get_state(uuid)
        if state is None:
            raise KeyError(f"No state found for UUID: {uuid}")
        
        synchronized_params = self.calculate_synchronized_params(instance, state)
        
        for param in synchronized_params:
            if param in state:
                setattr(instance, param, state[param])

    def synchronize_new_parameter(self, instance: Any, uuid: UUID, param_name: str) -> None:
        """
        This method should be used whenever a new parameter needs to be synchronized 
        between instances and state snapshots. It assumes the parameter is of a type 
        that should be synchronized, otherwise it will be overwritten by calculate_synchronized_params.
        """
        state = self.get_state(uuid)
        if state is None:
            raise KeyError(f"No state found for UUID: {uuid}")

        if not hasattr(instance, 'synchronized_params'):
            instance.synchronized_params = []
            instance.param_count = 0

        if param_name not in instance.synchronized_params:
            instance.synchronized_params.append(param_name)
            instance.param_count += 1
            
            # Add the new parameter to the state
            value = getattr(instance, param_name)
            state[param_name] = value
            
            # Update the state snapshot
            self.update_state(uuid, state)
        else:
            print(f"Warning: Parameter '{param_name}' is already synchronized.")

    @property
    def grid_size(self):
        return self._state['grid_size']

    @property
    def time(self):
        return self._state['time']

    @property
    def object_states(self):
        return self._state['object_states']

    def update_snapshot_with_items(self, items: List[Any]):
        # Verify all items are in the snapshot
        for item in items:
            if self.get_state(item.id) is None:
                raise KeyError(f"No state found for item: {item.id}")

        # Collect state changes
        state_changes = {}
        for item in items:
            state_changes[item.id] = item.update_state(self)

        # Update item states
        for item in items:
            self.update_state_params(item, item.id)

        # Process state changes
        for item_id, change_dict in state_changes.items():
            item = next(i for i in items if i.id == item_id)
            self.process_item_state_change_dict(item_id, change_dict, item)

    def update_snapshot_with_organisms(self, organisms: List[Any]):
        # Verify all organisms are in the snapshot
        for organism in organisms:
            if self.get_state(organism.id) is None:
                raise KeyError(f"No state found for organism: {organism.id}")

        # Collect state changes
        state_changes = {}
        for organism in organisms:
            state_changes[organism.id] = organism.update_state(self)

        # Update organism states
        for organism in organisms:
            self.update_state_params(organism, organism.id)

        # Process state changes
        for org_id, change_dict in state_changes.items():
            organism = next(org for org in organisms if org.id == org_id)
            self.process_organism_state_change_dict(org_id, change_dict, organism)


    def process_item_state_change_dict(self, item_id, change_dict, item):
        item_state = self.get_state(item_id)
        for key, value in change_dict.items():
            method_name = f"process_{key}"
            if hasattr(self, method_name):
                getattr(self, method_name)(item_id, value, item, item_state)
            else:
                print(f"Warning: No method to process {key}")

    def process_spawn_food(self, item_id, spawn_food: bool, item, item_state):
        if spawn_food:
            item.spawn_food(self)

    def process_organism_state_change_dict(self, org_id, change_dict, organism):
        org_state = self.get_state(org_id)
        for key, value in change_dict.items():
            method_name = f"process_{key}"
            if hasattr(self, method_name):
                getattr(self, method_name)(org_id, value, organism, org_state)
            else:
                print(f"Warning: No method to process {key}")

    def process_movement_vector(self, org_id: UUID, movement_vector, organism, org_state):
        dx, dy = movement_vector
        new_x = max(0, min(org_state['x'] + dx, self.sim_engine.world_width - 1))
        new_y = max(0, min(org_state['y'] + dy, self.sim_engine.world_height - 1))
        
        collision_objects = self.handle_collision(new_x, new_y, organism)
        if not collision_objects:
            org_state['x'] = new_x
            org_state['y'] = new_y
        else:
            for obj in collision_objects:
                # Check if the object is consumable
                if obj.consumable:
                    obj.consume(organism)
                    item_state = self.get_state(obj.id)
                    if item_state:
                        item_state['marked_for_deletion'] = True

    def process_attention_vector(self, org_id: UUID, attention_vector, organism, org_state):
        current_org_x, current_org_y = org_state['x'], org_state['y']
        current_attention_x, current_attention_y = org_state.get('attention_x', current_org_x), org_state.get('attention_y', current_org_y)
        dx, dy = attention_vector
        new_attention_x = max(0, min(current_attention_x + dx, self.sim_engine.world_width - 1))
        new_attention_y = max(0, min(current_attention_y + dy, self.sim_engine.world_height - 1))
        
        detection_radius = organism.detection_radius
        
        distance = self.sim_engine.calculate_distance(current_org_x, current_org_y, new_attention_x, new_attention_y)
        
        if distance <= detection_radius:
            org_state['attention_x'] = new_attention_x
            org_state['attention_y'] = new_attention_y
        else:
            angle = self.sim_engine.calculate_angle(current_org_x, current_org_y, new_attention_x, new_attention_y)
            constrained_x = current_org_x + detection_radius * math.cos(angle)
            constrained_y = current_org_y + detection_radius * math.sin(angle)
            org_state['attention_x'] = max(0, min(constrained_x, self.sim_engine.world_width - 1))
            org_state['attention_y'] = max(0, min(constrained_y, self.sim_engine.world_height - 1))
    
    def process_alive(self, org_id: UUID, alive: bool, organism, org_state):
        org_state['marked_for_deletion'] = not alive

    def process_spawn(self, org_id: UUID, should_spawn: bool, organism, org_state):
        if should_spawn:
            self.sim_engine.spawn_organism(organism, self)

    def handle_collision(self, x: float, y: float, organism) -> List[Any]:
        collision_objects = []
        for obj_id, obj_state in self.object_states.items():
            if (obj_id != organism.id and
                self.sim_engine.calculate_distance(x, y, obj_state['x'], obj_state['y']) <= self.sim_engine.collision_range and
                not obj_state.get('marked_for_deletion', False) and
                obj_state.get('collision', True)):
                
                obj = self.sim_engine.get_object_by_ID(obj_id)
                if obj:
                    collision_objects.append(obj)
        
        return collision_objects
