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
from collections import defaultdict


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
        ]
        
        # Simulation State Parameters
        self.x: float = initial_position[0]
        self.y: float = initial_position[1]
        self.attention_x: float = initial_position[0]
        self.attention_y: float = initial_position[1]
        self.marked_for_deletion: bool = False
        self.energy: float = 10.0
        self.nutrition: float = 0.0
      
        # Organismal Parameters
        self.movement_speed: float = 1.0
        self.attention_speed: float = 3.0 
        self.detection_radius: int = 100
        self.energy_consumption: float = 0.001
        self.nutrition_consumption: float = 0.0001
        self.max_nearest_items: int = 1
        self.nearest_item_params: int = 3  # distance, direction, reward

        # Neural Network Hyperparameters
        self.action_mapping: Dict[int, Action] = {action.value: action for action in Action}
        self.input_size: int = len(self.input_parameters) + self.max_nearest_items * self.nearest_item_params
        self.hidden_size: int = 16
        self.output_size: int = len(self.action_mapping)
        self.hidden_layers: int = 2
        self.learning_rate: float = 0.001
        self.gamma: float = 0.9
        self.training_steps: int = 0
        self.target_update: int = 100
        self.batch_size: int = 4
        self.capacity: int = 100000
        self.gradient_clip: float = 1.0
        self.epsilon: float = 1.0
        self.epsilon_min: float = 0.01
        self.epsilon_decay: float = 0.999
        self.boltzmann_temperature: float = 0.2

        # Training Metrics
        self.window_size = 100
        self.training_metrics = self.initialize_training_metrics()

        # Current Experience (Working) Memory
        self.current_experience: Optional[Experience] = None
        self.current_reward: float = 0.0
        self.nearest_item_ID: Optional[UUID] = None

        # Long-Term Memory
        self.item_memory: List[UUID] = []
        self.replay_buffer = PrioritizedExperienceReplay(capacity=self.capacity, batch_size=self.batch_size)

        # State Snapshot Synchronization Parameters
        self.synchronized_params: List[str] = []
        self.param_count: int = 0

        # RL Neural Network Initialization
        self.RL_algorithm: ReinforcementLearningAlgorithm = DQN(
            organism=self,
            action_mapping=self.action_mapping,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            hidden_layers=self.hidden_layers,
            learning_rate=self.learning_rate
        ) 

        # Thread-safe parameter diffs
        self.param_diffs = defaultdict(list)  # not being used currently, but these are for syncing params across threads
        self.param_diffs_lock = Lock()

    def set_input_parameters(self) -> None:
        """Initialize all input parameters to 0.0."""
        for param in self.input_parameters:
            self.__setattr__(param, 0.0)
    
    def create_metric_structure(self) -> Dict[str, Any]:
        return {
            "loss_history": deque(maxlen=self.window_size),
            "current_q_history": deque(maxlen=self.window_size),
            "expected_q_history": deque(maxlen=self.window_size),
            "loss_window_avg": 0.0,
            "current_q_window_avg": 0.0,
            "expected_q_window_avg": 0.0
        }

    def initialize_training_metrics(self) -> Dict[str, Any]:
        metrics = {}
        for action in self.action_mapping.values():
            metrics[action.name] = self.create_metric_structure()
        
        metrics["combined_averages"] = self.create_metric_structure()
        metrics["reward"] = {
            "reward_history": deque(maxlen=self.window_size),
            "reward_window_avg": 0.0
        }
        return metrics

    def get_internal_state(self, external_state: StateSnapshot) -> torch.Tensor:
        """Compute and return the internal state of the organism."""
        nearest_items_vector = self.calculate_nearest_items_vector(external_state)
        self.calculate_attention_item_parameters(external_state)
        self.calculate_organism_attention_parameters()

        state = [getattr(self, param) for param in self.input_parameters]
        state.extend(nearest_items_vector)

        return torch.tensor(state, dtype=torch.float32)

    def calculate_nearest_items_vector(self, external_state: StateSnapshot) -> List[float]:
        """Calculate and return a vector representing the nearest items."""
        nearest_items_vector = []
        nearest_item_ids = self.sim_engine.get_nearest_items(
            self.x, self.y, 
            self.max_nearest_items, 
            self.detection_radius,
            external_state,
            item_type='food',
            return_IDs=True
        )

        self.nearest_item_ID = nearest_item_ids[0] if nearest_item_ids else None
        self.update_item_memory(nearest_item_ids)

        if not nearest_item_ids:
            nearest_item_ids = self.get_item_from_memory(self.x, self.y, external_state)
            if nearest_item_ids:
                self.nearest_item_ID = nearest_item_ids[0]

        for item_id in nearest_item_ids:
            item_state = external_state.get_state(item_id)
            if item_state:
                distance_org_item = self.sim_engine.calculate_distance(self.x, self.y, item_state['x'], item_state['y'], normalize_to_viewport=True)
                direction_org_item = self.sim_engine.calculate_angle(self.x, self.y, item_state['x'], item_state['y'], normalize=True)
                nearest_items_vector.extend([distance_org_item, direction_org_item, item_state['reward']])

        padding_length = (self.max_nearest_items - len(nearest_item_ids)) * self.nearest_item_params
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

    def update_state(self, external_state: StateSnapshot) -> Dict[str, Any]:
        """Update the organism's state and return the changes to be applied externally."""
        internal_state = self.get_internal_state(external_state)
        action_index = self.RL_algorithm.select_action(internal_state)
        attention_vector = self.update_attention_point(action_index)
        movement_vector = self.move(attention_vector)
        
        external_state_change = {
            'movement_vector': movement_vector,
            'attention_vector': attention_vector,
        }
        
        metabolism_changes = self.update_metabolism()
        
        external_state_change.update(metabolism_changes)
        
        self.current_experience = Experience(
            state=internal_state,
            action=action_index,
            reward=None,  # Will be set in apply_state
            next_state=None  # Will be set in apply_state
        )
        return external_state_change

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
                reward = proximity_reward + direction_reward
                
                if new_distance <= 2:
                    reward += 1.0
                    
                    attention_movement = self.sim_engine.calculate_distance(old_attention_x, old_attention_y, new_attention_x, new_attention_y)
                    if attention_movement < 0.1:
                        reward += 1.0
            else:
                nearest_item_id = None
                reward = -0.666
        else:
            reward = -0.666
        
        reward += self.current_reward
        self.current_reward = 0.0
        
        return reward
    
    def calculate_direction_reward(self, angle_diff: float) -> float:
        """Calculate the direction reward based on the angle difference."""
        normalized_angle = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        reward = 0.5 * math.cos(normalized_angle)
        return reward

    def apply_state(self, old_state: StateSnapshot, new_state: StateSnapshot) -> None:
        """Apply the new state, calculate rewards, and trigger learning."""
        total_reward = self.calculate_rewards(old_state, new_state)
        new_internal_state = self.get_internal_state(new_state)
        
        if self.current_experience:
            updated_experience = Experience(
                state=self.current_experience.state,
                action=self.current_experience.action,
                reward=total_reward,
                next_state=new_internal_state
            )
            self.replay_buffer.add(updated_experience)
        
        self.current_experience = None
        
        # Queue the learning task instead of starting a new thread
        self.RL_algorithm.queue_learn(new_state, total_reward)
    
    def record_training_metrics(self, metrics: Dict[str, Any], total_reward: float) -> None:
        """Record and update training metrics including loss, Q-values, and rewards."""
        combined_loss = []
        combined_current_q = []
        combined_expected_q = []

        for action_name, action_metrics in metrics.items():
            if action_metrics["loss"]:
                self.training_metrics[action_name]["loss_history"].extend(action_metrics["loss"])
                self.training_metrics[action_name]["current_q_history"].extend(action_metrics["current_q"])
                self.training_metrics[action_name]["expected_q_history"].extend(action_metrics["expected_q"])
                
                self.training_metrics[action_name]["loss_window_avg"] = sum(self.training_metrics[action_name]["loss_history"]) / len(self.training_metrics[action_name]["loss_history"])
                self.training_metrics[action_name]["current_q_window_avg"] = sum(self.training_metrics[action_name]["current_q_history"]) / len(self.training_metrics[action_name]["current_q_history"])
                self.training_metrics[action_name]["expected_q_window_avg"] = sum(self.training_metrics[action_name]["expected_q_history"]) / len(self.training_metrics[action_name]["expected_q_history"])

                combined_loss.extend(action_metrics["loss"])
                combined_current_q.extend(action_metrics["current_q"])
                combined_expected_q.extend(action_metrics["expected_q"])

        if combined_loss:
            avg_loss = sum(combined_loss) / len(combined_loss)
            avg_current_q = sum(combined_current_q) / len(combined_current_q)
            avg_expected_q = sum(combined_expected_q) / len(combined_expected_q)

            self.training_metrics["combined_averages"]["loss_history"].append(avg_loss)
            self.training_metrics["combined_averages"]["current_q_history"].append(avg_current_q)
            self.training_metrics["combined_averages"]["expected_q_history"].append(avg_expected_q)
            
            self.training_metrics["combined_averages"]["loss_window_avg"] = sum(self.training_metrics["combined_averages"]["loss_history"]) / len(self.training_metrics["combined_averages"]["loss_history"])
            self.training_metrics["combined_averages"]["current_q_window_avg"] = sum(self.training_metrics["combined_averages"]["current_q_history"]) / len(self.training_metrics["combined_averages"]["current_q_history"])
            self.training_metrics["combined_averages"]["expected_q_window_avg"] = sum(self.training_metrics["combined_averages"]["expected_q_history"]) / len(self.training_metrics["combined_averages"]["expected_q_history"])

        # Record the total reward
        self.training_metrics["reward"]["reward_history"].append(total_reward)
        self.training_metrics["reward"]["reward_window_avg"] = sum(self.training_metrics["reward"]["reward_history"]) / len(self.training_metrics["reward"]["reward_history"])

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
        if self.energy > 100 and self.nutrition > 100:
            self.energy -= 50
            self.nutrition -= 30
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
