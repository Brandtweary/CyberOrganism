from typing import Dict, Any, Optional, List, Tuple, Union
from uuid import UUID
import math
import copy
from shared.enums import ObjectType
from shared.summary_logger import summary_logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from shared.custom_profiler import profiler
import gc


class StateSnapshot:
    def __init__(self, SimulationEngine, start_time: float, grid_size: int):
        self._state = {
            'object_states': {},
            'grid_size': grid_size,
            'time': start_time,
            'start_time': start_time,
            'elapsed_time': 0.0
        }
        self.sim_engine = SimulationEngine
        self.use_threading = False
        if self.use_threading:
            self.thread_pool = ThreadPoolExecutor(max_workers=64)

    @profiler.profile("clone_state_snapshot")
    def clone_state_snapshot(self) -> 'StateSnapshot':
        with profiler.profile_section("clone_state_snapshot", "create_snapshot"):
            new_snapshot = self.__class__(self.sim_engine, self._state['start_time'], self._state['grid_size'])
        
        with profiler.profile_section("clone_state_snapshot", "copy_state"):
            # Shallow copy the top level dictionary
            new_snapshot._state = self._state.copy()
            
            # Deep copy only the 'object_states' dictionary
            new_snapshot._state['object_states'] = {
                uuid: state.copy() for uuid, state in self._state['object_states'].items()
            }

        return new_snapshot
        
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
        self._state['object_states'][uuid].update(state)

    def remove_state(self, uuid: UUID):
        self._state['object_states'].pop(uuid, None)

    def update_time(self, new_time: float):
        self._state['time'] = new_time
        self._state['elapsed_time'] = new_time - self._state['start_time']

    def update_grid_size(self, new_grid_size: int):
        self._state['grid_size'] = new_grid_size

    def calculate_synchronized_params(self, instance: Any, state: Dict[str, Any], initial_calc: bool = False) -> List[str]:
        '''
        Do NOT synchronize mutable params. Instance params are only shallow copied when the snapshot is cloned.
        We had been deepcopying before but it was slow and also causing garbage collection issues. 
        '''
        if not hasattr(instance, 'synchronized_params') or not hasattr(instance, 'param_count') or len(state) != instance.param_count or initial_calc:
            synchronized_params = [
                param for param, value in instance.__dict__.items()
                if isinstance(value, (int, float, str, bool, tuple, UUID)) or 
                (param.endswith('_ID') and value is None)  # immutable params only
            ]
            instance.synchronized_params = synchronized_params
            instance.param_count = len(synchronized_params)
        return instance.synchronized_params

    def update_state_params(self, instance: Any, uuid: UUID):
        state = self.get_state(uuid)
        initial_calc = False
        if state is None:
            state = {}
            self.add_state(uuid, state)
            initial_calc = True
        
        synchronized_params = self.calculate_synchronized_params(instance, state, initial_calc)
        
        for param in synchronized_params:
            value = getattr(instance, param)
            if param not in state or state[param] != value:
                state[param] = value

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
            
            value = getattr(instance, param_name)
            state[param_name] = value
        else:
            summary_logger.warning(f"Warning: Parameter '{param_name}' is already synchronized.")

    @property
    def grid_size(self):
        return self._state['grid_size']

    @property
    def time(self):
        return self._state['time']
    
    @property
    def elapsed_time(self):
        return self._state['elapsed_time']

    @property
    def object_states(self):
        return self._state['object_states']
    
    def synchronize_param_diffs(self, objects: List[Any]):
        for obj in objects:
            if hasattr(obj, 'get_and_clear_param_diffs'):
                diffs = obj.get_and_clear_param_diffs()
                for param_name, diff_list in diffs.items():
                    if len(diff_list) == 1:
                        setattr(obj, param_name, diff_list[0])
                    else:
                        original_value = getattr(obj, param_name)
                        if isinstance(original_value, (int, float)):
                            total_diff = sum(diff - original_value for diff in diff_list if isinstance(diff, (int, float)))
                            new_value = original_value + total_diff
                        else:
                            new_value = diff_list[-1]
                        setattr(obj, param_name, new_value)

    def update_snapshot_with_objects(self, objects: List[Any]):
        if self.use_threading:
            self._update_snapshot_with_objects_threaded(objects)
        else:
            self._update_snapshot_with_objects_sequential(objects)

    @profiler.profile("_update_snapshot_with_objects_threaded")
    def _update_snapshot_with_objects_threaded(self, objects):  
        self.synchronize_param_diffs(objects)
        self.batch_state_preparation(objects)

        future_to_obj = {
            self.thread_pool.submit(obj.update_state, self): obj 
            for obj in objects
        }
        state_changes = {}
        for future in as_completed(future_to_obj):
            obj = future_to_obj[future]
            try:
                state_changes[obj.id] = future.result()
            except Exception as exc:
                summary_logger.error(f"Object {obj.id} generated an exception: {exc}")

        with profiler.profile_section("_update_snapshot_with_objects_threaded", "post_processing"):
            for obj in objects:
                self.update_state_params(obj, obj.id)

            for obj_id, change_dict in state_changes.items():
                obj = next(o for o in objects if o.id == obj_id)
                self.process_state_change_dict(obj_id, change_dict, obj, self.get_state(obj_id))

    @profiler.profile("_update_snapshot_with_objects_sequential")
    def _update_snapshot_with_objects_sequential(self, objects):
        with profiler.profile_section("_update_snapshot_with_objects_sequential", "synchronize_param_diffs"):
            self.synchronize_param_diffs(objects)

        with profiler.profile_section("_update_snapshot_with_objects_sequential", "batch_state_preparation"):
            self.batch_state_preparation(objects)

        with profiler.profile_section("_update_snapshot_with_objects_sequential", "update_states"): 
            state_changes = {}
            for obj in objects:
                state_changes[obj.id] = obj.update_state(self)

        with profiler.profile_section("_update_snapshot_with_objects_sequential", "update_params"):
            for obj in objects:
                self.update_state_params(obj, obj.id)

        with profiler.profile_section("_update_snapshot_with_objects_sequential", "process_state_changes"):
            for obj_id, change_dict in state_changes.items():
                obj = next(o for o in objects if o.id == obj_id)
                self.process_state_change_dict(obj_id, change_dict, obj, self.get_state(obj_id))

    def batch_state_preparation(self, objects):

        # Get nearest items for organisms
        organisms = [obj for obj in objects if obj.type == ObjectType.ORGANISM]
        for organism in organisms:
            org_state = self.get_state(organism.id)
            nearest_item_ids = self.sim_engine.get_nearest_items(
                org_state['x'], org_state['y'], 
                organism.current_nearest_items,
                organism.detection_radius,
                self,
                item_type='food',
                return_IDs=True
            )
            organism.nearest_item_ids = nearest_item_ids

    def process_state_change_dict(self, obj_id: UUID, change_dict: Dict[str, Any], obj: Any, obj_state: Dict[str, Any]):
        for key, value in change_dict.items():
            method_name = f"process_{key}"
            if hasattr(self, method_name):
                getattr(self, method_name)(obj_id, value, obj, obj_state)
            else:
                summary_logger.warning(f"Warning: No method to process {key} for object {obj_id}")
    
    def process_spawn_food(self, item_id, spawn_food: bool, item, item_state):
        if spawn_food:
            item.spawn_food(self)

    def process_movement_vector(self, org_id: UUID, movement_vector, organism, org_state):
        dx, dy = movement_vector
        new_x = max(0, min(org_state['x'] + dx, self.sim_engine.world_width - 1))
        new_y = max(0, min(org_state['y'] + dy, self.sim_engine.world_height - 1))
        
        collision_objects = self.handle_collision(new_x, new_y, organism)

        blocking_objects = [
            obj for obj in collision_objects 
            if not obj.consumable and 
            self.get_state(obj.id)['x'] == new_x and 
            self.get_state(obj.id)['y'] == new_y
        ]
        if not blocking_objects:
            org_state['x'] = new_x
            org_state['y'] = new_y

        for obj in collision_objects:
            if obj.consumable:
                obj.consume(org_state)
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
            if len(self.sim_engine.organisms) < self.sim_engine.max_zoomorphs:
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
