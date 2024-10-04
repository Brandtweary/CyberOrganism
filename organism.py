import torch
import math
from collections import deque
from typing import Dict, Any, Tuple, List, Optional
from uuid import UUID, uuid4
from state_snapshot import StateSnapshot, ObjectType
from prioritized_experience_replay import PrioritizedExperienceReplay, Experience
from enums import Action
from RL_algorithm import ReinforcementLearningAlgorithm
from dqn import DQN
from threading import Lock
from collections import defaultdict, Counter
import random
from custom_profiler import profiler


class Organism:
    """
    Represents a simulated agent that uses deep reinforcement learning to select actions and learn.
    
    This class manages state representations, tracks hyperparameters, synchronizes with the simulation engine,
    and handles its own metabolism. The specific learning details are abstracted into the RL Algorithm class.
    """

    def __init__(self, SimulationEngine: Any, initial_position: Tuple[int, int]) -> None:
        """Initialize the organism with given simulation engine and starting position."""
        # Static Parameters
        self.id: UUID = uuid4()
        self.sim_engine = SimulationEngine
        self.type: ObjectType = ObjectType.ORGANISM
        self.collision: bool = True
        self.consumable: bool = False

        # Neural Network Input Parameters
        self.input_parameters: List[str] = [
            'attention_item_distance',
            'attention_item_direction',
            'organism_attention_distance',
            'organism_attention_direction'
        ]
        self.set_input_parameters()  # all default to 0.0

        # HUD Display Parameters
        self.display_parameters: List[str] = [
            'energy',
            'nutrition',
            'epsilon',
            'current_nearest_items',
            'proximity_reward_weight',
            'direction_reward_weight',
            'focus_reward_weight',
            'replay_buffer_size',
            'target_update_counter',
            'inference_update_counter'
        ]
        
        # Simulation State Parameters
        self.x: float = initial_position[0]
        self.y: float = initial_position[1]
        self.attention_x: float = initial_position[0]
        self.attention_y: float = initial_position[1]
        self.marked_for_deletion: bool = False
        self.energy: float = 10.0
        self.nutrition: float = 0.0
        self.just_spawned: bool = True
        self.frame_skip = 4  # Skip 3 out of 4 frames
        self.frame_skip_counter = random.randint(0, self.frame_skip - 1)  # Initialize randomly
      
        # Organismal Parameters
        self.movement_speed: float = 1.0
        self.attention_speed: float = 3.0 
        self.detection_radius: int = 200
        self.energy_consumption: float = 0.002
        self.nutrition_consumption: float = 0.0001
        self.start_nearest_items: int = 1
        self.max_nearest_items: int = 1
        self.nearest_item_params: int = 3  # distance, direction, reward
        self.current_nearest_items: int = self.start_nearest_items
        self.nearest_items_curriculum_period: float = 30 * 60  # 30 minutes in seconds
        self.proximity_reward_weight: float = 2.0
        self.direction_reward_weight: float = 1.0
        self.focus_reward_weight: float = 1.0
        self.target_update_counter: int = 0
        self.inference_update_counter: int = 0

        # Neural Network Hyperparameters
        self.action_mapping: Dict[int, Action] = {action.value: action for action in Action}
        self.input_size: int = len(self.input_parameters) + self.max_nearest_items * self.nearest_item_params
        self.hidden_size: int = 64
        self.output_size: int = len(self.action_mapping)
        self.hidden_layers: int = 2
        self.learning_rate: float = 0.001
        self.gamma: float = 0.9
        self.target_update: int = 50
        self.inference_update: int = 1
        self.batch_size: int = 4
        self.capacity: int = 10000
        self.gradient_clip: float = 1.0
        self.epsilon: float = 0.8
        self.epsilon_min: float = 0.01
        self.epsilon_decay: float = 0.999
        self.boltzmann_temperature: float = 0.2

        # Training Metrics
        self.average_reward = 0.0
        self.average_loss = 0.0
        self.average_q_value = 0.0
        self.reward_history = deque(maxlen=100)
        self.action_history = deque(maxlen=1000)
        self.action_distribution: List[Tuple[str, float]] = []
        self.action_counter = Counter()
        self.last_action_index = None

        # Current Experience (Working) Memory
        self.current_experience: Optional[Experience] = None
        self.current_reward: float = 0.0
        self.nearest_item_ID: Optional[UUID] = None
        self.nearest_item_ids: List[UUID] = []

        # Long-Term Memory
        self.item_memory: List[UUID] = []
        self.replay_buffer = PrioritizedExperienceReplay(capacity=self.capacity, batch_size=self.batch_size)
        self.replay_buffer_size: int = 0

        # State Snapshot Synchronization Parameters
        self.synchronized_params: List[str] = []
        self.param_count: int = 0

        # RL Neural Network Initialization
        network_params = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate
        }

        self.RL_algorithm: ReinforcementLearningAlgorithm = DQN(
            organism=self,
            action_mapping=self.action_mapping,
            network_params=network_params
        )

        # Thread-safe parameter diffs
        self.param_diffs = defaultdict(list)  # not being used currently, but these are for syncing params across threads
        self.param_diffs_lock = Lock()

    def set_input_parameters(self) -> None:
        """Initialize all input parameters to 0.0."""
        for param in self.input_parameters:
            self.__setattr__(param, 0.0)

    def adjust_input_sensors(self, external_state: StateSnapshot) -> None:
        """Adjust the number of nearest items based on elapsed time."""
        elapsed_time = external_state.elapsed_time
        
        if elapsed_time <= self.nearest_items_curriculum_period:
            progress = min(elapsed_time / self.nearest_items_curriculum_period, 1.0)
            self.current_nearest_items = self.start_nearest_items + math.floor(progress * (self.max_nearest_items - self.start_nearest_items))
        else:
            self.current_nearest_items = self.max_nearest_items

    @profiler.profile("get_internal_state")
    def get_internal_state(self, external_state: StateSnapshot) -> torch.Tensor:
        """Compute and return the internal state of the organism."""
        self.adjust_input_sensors(external_state)
        with profiler.profile_section("get_internal_state", "calculate_nearest_items_vector"):
            nearest_items_vector = self.calculate_nearest_items_vector(external_state)
        with profiler.profile_section("get_internal_state", "calculate_attention_item_parameters"):
            self.calculate_attention_item_parameters(external_state)
        with profiler.profile_section("get_internal_state", "calculate_organism_attention_parameters"):
            self.calculate_organism_attention_parameters()

        state = [getattr(self, param) for param in self.input_parameters]
        state.extend(nearest_items_vector)

        return torch.tensor(state, dtype=torch.float32)

    @profiler.profile("calculate_nearest_items_vector")
    def calculate_nearest_items_vector(self, external_state: StateSnapshot) -> List[float]:
        """Calculate and return a vector representing the nearest items."""
        nearest_items_vector = []

        with profiler.profile_section("calculate_nearest_items_vector", "item_memory"):
            self.nearest_item_ID = self.nearest_item_ids[0] if self.nearest_item_ids else None  # these are now set in batch_state_preparation
            self.update_item_memory(self.nearest_item_ids)

            if not self.nearest_item_ids:
                self.nearest_item_ids = self.get_item_from_memory(self.x, self.y, external_state)
                if self.nearest_item_ids:
                    self.nearest_item_ID = self.nearest_item_ids[0]

        with profiler.profile_section("calculate_nearest_items_vector", "create_nearest_item_vectors"):
            for item_id in self.nearest_item_ids:
                item_state = external_state.get_state(item_id)
                if item_state:
                    distance_org_item = self.sim_engine.calculate_distance(self.x, self.y, item_state['x'], item_state['y'], normalize_to_viewport=True)
                    direction_org_item = self.sim_engine.calculate_angle(self.x, self.y, item_state['x'], item_state['y'], normalize=True)
                    nearest_items_vector.extend([distance_org_item, direction_org_item, item_state['reward']])

            padding_length = (self.max_nearest_items - len(self.nearest_item_ids)) * self.nearest_item_params
            nearest_items_vector.extend([0.0] * padding_length)

        return nearest_items_vector

    def calculate_attention_item_parameters(self, external_state: StateSnapshot) -> None:
        """Calculate the distance and direction to the nearest item from the attention point."""
        if self.nearest_item_ID:
            item_state = external_state.get_state(self.nearest_item_ID)
            if item_state:
                self.attention_item_distance = self.sim_engine.calculate_distance(self.attention_x, self.attention_y, item_state['x'], item_state['y'], normalize_to_viewport=True)
                self.attention_item_direction = self.sim_engine.calculate_angle(self.attention_x, self.attention_y, item_state['x'], item_state['y'], normalize=True)
            else:
                self.attention_item_distance = 0.0
                self.attention_item_direction = 0.0
        else:
            self.attention_item_distance = 0.0
            self.attention_item_direction = 0.0

    def calculate_organism_attention_parameters(self) -> None:
        """Calculate the distance and direction from the organism to its attention point."""
        self.organism_attention_distance = self.sim_engine.calculate_distance(self.x, self.y, self.attention_x, self.attention_y, normalize_to_viewport=True)
        self.organism_attention_direction = self.sim_engine.calculate_angle(self.x, self.y, self.attention_x, self.attention_y, normalize=True)

    def update_item_memory(self, nearest_item_ids: List[UUID]) -> None:
        """Update the item_memory list with new unique IDs."""
        for item_id in nearest_item_ids:
            if item_id not in self.item_memory:
                self.item_memory.append(item_id)

    def get_item_from_memory(self, organism_x: float, organism_y: float, external_state: StateSnapshot) -> List[Optional[UUID]]:
        """Retrieve the nearest remembered item that still exists."""
        memory_copy = self.item_memory.copy()
        nearest_item = None
        nearest_distance = float('inf')

        for item_id in memory_copy:
            item_state = external_state.get_state(item_id)
            if item_state:
                distance = self.sim_engine.calculate_distance(organism_x, organism_y, item_state['x'], item_state['y'])
                if distance < nearest_distance:
                    nearest_item = item_id
                    nearest_distance = distance
            else:
                self.item_memory.remove(item_id)

        return [nearest_item] if nearest_item else []

    def update_attention_point(self, action_index: int) -> Tuple[float, float]:
        """Update the attention point based on the given action."""
        dx, dy = 0, 0
        action = self.action_mapping[action_index]  # Map action index to Action enum
        if action == Action.UP:
            dy = -self.attention_speed
        elif action == Action.DOWN:
            dy = self.attention_speed
        elif action == Action.LEFT:
            dx = -self.attention_speed
        elif action == Action.RIGHT:
            dx = self.attention_speed
        elif action == Action.NO_MOVE:
            return 0, 0

        return dx, dy

    def move(self, attention_vector: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate the movement vector based on the attention vector."""
        current_attention_x = self.attention_x
        current_attention_y = self.attention_y
        
        theoretical_attention_x = current_attention_x + attention_vector[0]
        theoretical_attention_y = current_attention_y + attention_vector[1]
        
        dx = theoretical_attention_x - self.x
        dy = theoretical_attention_y - self.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > 0:
            direction_x = dx / distance
            direction_y = dy / distance
            
            move_x = direction_x * self.movement_speed
            move_y = direction_y * self.movement_speed
            
            return move_x, move_y
        else:
            return 0, 0

    @profiler.profile("update_state")
    def update_state(self, external_state: StateSnapshot) -> Dict[str, Any]:
        """Update the organism's state and return the changes to be applied externally."""
        self.just_spawned = False
        self.frame_skip_counter = (self.frame_skip_counter + 1) % self.frame_skip

        if self.frame_skip_counter == 0:
            with profiler.profile_section("update_state", "get_internal_state"):
                internal_state = self.get_internal_state(external_state)
            
            with profiler.profile_section("update_state", "select_action"):
                action_index = self.RL_algorithm.select_action(internal_state)
        else:
            action_index = self.last_action_index if self.last_action_index is not None else random.choice(list(self.action_mapping.keys()))

        with profiler.profile_section("update_state", "calculate_action_distribution"):
            self.calculate_action_distribution(action_index)

        with profiler.profile_section("update_state", "action_processing"):
            attention_vector = self.update_attention_point(action_index)
            movement_vector = self.move(attention_vector)
            
            external_state_change = {
                'movement_vector': movement_vector,
                'attention_vector': attention_vector,
            }
            
            metabolism_changes = self.update_metabolism()
            external_state_change.update(metabolism_changes)
            
            if self.frame_skip_counter == 0:
                # Only store experience on active frames
                self.current_experience = Experience(
                    state=internal_state,
                    action=action_index,
                    reward=None, 
                    next_state=None
                )
        
        return external_state_change
    
    def calculate_action_distribution(self, action_index: int) -> None:
        """Calculate the action distribution based on the action index."""
        action_name = self.action_mapping[action_index].name
        
        # Update action history and counter
        if len(self.action_history) == 1000:
            old_action = self.action_history[0]
            self.action_counter[self.action_mapping[old_action].name] -= 1
        
        self.action_history.append(action_index)
        self.action_counter[action_name] += 1
        
        # Update last action index
        self.last_action_index = action_index

        # Update distribution every 10 actions (or another suitable interval)
        if len(self.action_history) % 10 == 0:
            total_actions = len(self.action_history)
            self.action_distribution = [
                (action.name, self.action_counter.get(action.name, 0) / total_actions)
                for action in self.action_mapping.values()
            ]

    def calculate_rewards(self, old_state: StateSnapshot, new_state: StateSnapshot) -> float:
        """Calculate the reward based on the old and new states."""
        old_organism_state = old_state.get_state(self.id)
        if old_organism_state is None:
            raise ValueError(f"Old organism state not found for ID: {self.id}")
        
        old_attention_x, old_attention_y = old_organism_state['attention_x'], old_organism_state['attention_y']
        new_attention_x, new_attention_y = self.attention_x, self.attention_y
        
        nearest_item_id = self.nearest_item_ID
        
        if nearest_item_id:
            item_state = new_state.get_state(nearest_item_id)
            if item_state:
                item_x, item_y = item_state['x'], item_state['y']
                new_distance = self.sim_engine.calculate_distance(new_attention_x, new_attention_y, item_x, item_y)
                normalized_distance = self.sim_engine.calculate_distance(new_attention_x, new_attention_y, item_x, item_y, normalize_to_viewport=True)
                proximity_reward = 0.5 * (1 - 2 * normalized_distance)
                attention_delta_x = new_attention_x - old_attention_x
                attention_delta_y = new_attention_y - old_attention_y

                if attention_delta_x == 0 and attention_delta_y == 0:
                    movement_angle = 0
                else:
                    movement_angle = math.atan2(attention_delta_y, attention_delta_x)

                target_delta_x = item_x - old_attention_x
                target_delta_y = item_y - old_attention_y
                target_angle = math.atan2(target_delta_y, target_delta_x)
                angle_diff = math.atan2(math.sin(target_angle - movement_angle), math.cos(target_angle - movement_angle))
                direction_reward = self.calculate_direction_reward(angle_diff)
                reward = (proximity_reward * self.proximity_reward_weight) + (direction_reward * self.direction_reward_weight)
                
                if new_distance <= 2:
                    reward += (1.0 * self.focus_reward_weight)
                    
                    attention_movement = self.sim_engine.calculate_distance(old_attention_x, old_attention_y, new_attention_x, new_attention_y)
                    if attention_movement < 0.1:
                        reward += (1.0 * self.focus_reward_weight)
            else:
                nearest_item_id = None
                reward = -0.666
        else:
            reward = -0.666
        
        reward += self.current_reward
        self.current_reward = 0.0

        self.adjust_reward_weights()
        self.reward_history.append(reward)
        self.average_reward = sum(self.reward_history) / len(self.reward_history)
        
        return reward
    
    def adjust_reward_weights(self) -> None:
        self.proximity_reward_weight *= 1 - 1e-4
        self.direction_reward_weight *= 1 - 1e-5
        self.focus_reward_weight *= 1 - 1e-6
    
    def calculate_direction_reward(self, angle_diff: float) -> float:
        """Calculate the direction reward based on the angle difference."""
        normalized_angle = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        reward = 0.5 * math.cos(normalized_angle)
        return reward

    @profiler.profile("apply_state")
    def apply_state(self, old_state: StateSnapshot, new_state: StateSnapshot) -> None:
        """Apply the new state, calculate rewards, and potentially queue learning."""
        if self.just_spawned:
            return  # organism is not in the old state, so the following code would throw an error

        if self.frame_skip_counter == 0:
            with profiler.profile_section("apply_state", "internal_processing"):
                total_reward = self.calculate_rewards(old_state, new_state)
                new_internal_state = self.get_internal_state(new_state)
            
            
            with profiler.profile_section("apply_state", "replay_buffer"):
                if self.current_experience:
                    updated_experience = Experience(
                        state=self.current_experience.state,
                        action=self.current_experience.action,
                        reward=total_reward,
                        next_state=new_internal_state
                    )
                    self.replay_buffer.add(updated_experience)
                    self.replay_buffer_size = self.replay_buffer.get_tree_size()

            self.current_experience = None
            
            with profiler.profile_section("apply_state", "queue_learn_conditionally"):
                self.queue_learn_conditionally(new_state)

    @profiler.profile("queue_learn_conditionally")
    def queue_learn_conditionally(self, new_state: StateSnapshot) -> None:
        """Conditionally queue a learning task if the replay buffer can be sampled."""
        if self.replay_buffer.can_sample(): 
            with profiler.profile_section("queue_learn_conditionally", "sample"):
                experiences = self.replay_buffer.sample()
            with profiler.profile_section("queue_learn_conditionally", "queue_learn"):
                organism_state = new_state.get_state(self.id).copy()
                self.RL_algorithm.queue_learn(organism_state, experiences)
    
    def update_metabolism(self) -> Dict[str, Any]:
        """Update the organism's energy and nutrition levels, and check for reproduction."""
        self.energy -= self.energy_consumption
        self.nutrition = max(0, self.nutrition - self.nutrition_consumption)
        
        should_spawn = self.handle_reproduction()
        
        return {
            'alive': self.energy > 0,
            'spawn': should_spawn
        }

    def handle_reproduction(self) -> bool:
        """Check if the organism can reproduce and update its energy and nutrition accordingly."""
        if self.energy > 100 and self.nutrition > 30 and len(self.sim_engine.organisms) < self.sim_engine.max_zoomorphs:
            self.energy -= 80
            self.nutrition -= 20
            return True
        return False

    def add_param_diff(self, param_name: str, diff_value: Any):
        with self.param_diffs_lock:
            self.param_diffs[param_name].append(diff_value)

    def get_and_clear_param_diffs(self):
        with self.param_diffs_lock:
            diffs = dict(self.param_diffs)
            self.param_diffs.clear()
        return diffs
