from typing import Dict, Any, Optional, List
from uuid import UUID
import math
import copy


class StateSnapshot:
    def __init__(self, matrika, current_time: float, grid_size: int):
        self._state = {
            'object_states': {},
            'grid_size': grid_size,
            'time': current_time
        }
        self.matrika = matrika

    def clone_state_snapshot(self) -> 'StateSnapshot':
        return copy.deepcopy(self)

    def get_state(self, uuid: UUID) -> Optional[Dict[str, Any]]:
        return self._state['object_states'].get(uuid)
    
    def get_objects_in_snapshot(self):
        return self._state['object_states'].items()
    
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

    def update_state_params(self, instance: Any, uuid: UUID):
        state = self.get_state(uuid)
        if state is None:
            raise KeyError(f"No state found for UUID: {uuid}")
        
        for param, value in instance.__dict__.items():
            if param not in state or state[param] != value:
                state[param] = value
        
        self.update_state(uuid, state)

    def apply_state_params(self, instance: Any, uuid: UUID):
        state = self.get_state(uuid)
        if state is None:
            raise KeyError(f"No state found for UUID: {uuid}")
        
        for param, value in state.items():
            setattr(instance, param, value)

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
            self.process_organism_state_change_dict(org_id, change_dict, organism)

    def process_item_state_change_dict(self, item_id, change_dict, item):
        # Currently, this method does nothing as items don't produce state changes
        pass

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
        new_x = max(0, min(org_state['x'] + dx, self.matrika.world_width - 1))
        new_y = max(0, min(org_state['y'] + dy, self.matrika.world_height - 1))
        
        collision_objects = self.handle_collision(new_x, new_y, organism)
        if not collision_objects:
            org_state['x'] = new_x
            org_state['y'] = new_y
        else:
            for obj in collision_objects:
                if isinstance(obj, self.matrika.Food):  # Assuming Food is a class in Matrika
                    obj.consume(organism)
                    item_state = self.get_state(obj.id)
                    if item_state:
                        item_state['marked_for_deletion'] = True

    def process_attention_vector(self, org_id: UUID, attention_vector, organism, org_state):
        current_org_x, current_org_y = org_state['x'], org_state['y']
        current_attention_x, current_attention_y = org_state.get('attention_point', (current_org_x, current_org_y))
        dx, dy = attention_vector
        new_attention_x = max(0, min(current_attention_x + dx, self.matrika.world_width - 1))
        new_attention_y = max(0, min(current_attention_y + dy, self.matrika.world_height - 1))
        
        detection_radius = organism.detection_radius
        
        distance = self.matrika.calculate_distance(current_org_x, current_org_y, new_attention_x, new_attention_y)
        
        if distance <= detection_radius:
            org_state['attention_point'] = (new_attention_x, new_attention_y)
        else:
            angle = self.matrika.calculate_angle(current_org_x, current_org_y, new_attention_x, new_attention_y)
            constrained_x = current_org_x + detection_radius * math.cos(angle)
            constrained_y = current_org_y + detection_radius * math.sin(angle)
            constrained_x = max(0, min(constrained_x, self.matrika.world_width - 1))
            constrained_y = max(0, min(constrained_y, self.matrika.world_height - 1))
            org_state['attention_point'] = (constrained_x, constrained_y)
    
    def process_alive(self, org_id: UUID, alive: bool, organism, org_state):
        org_state['marked_for_deletion'] = not alive

    def process_spawn(self, org_id: UUID, should_spawn: bool, organism, org_state):
        if should_spawn:
            self.matrika.spawn_organism(organism, self)

    def handle_collision(self, x: float, y: float, organism) -> List[Any]:
        collision_objects = []
        for item_id, item_state in self.object_states.items():
            if (self.matrika.calculate_distance(x, y, item_state['x'], item_state['y']) <= self.matrika.collision_range
                and not item_state.get('marked_for_deletion', False)):
                item = self.matrika.get_item_by_ID(item_id)
                if item:
                    collision_objects.append(item)
        
        for org_id, org_state in self.object_states.items():
            if (org_id != organism.id 
                and self.matrika.calculate_distance(x, y, org_state['x'], org_state['y']) <= self.matrika.collision_range
                and not org_state.get('marked_for_deletion', False)):
                collision_org = self.matrika.get_organism_by_ID(org_id)
                if collision_org:
                    collision_objects.append(collision_org)
        
        return collision_objects