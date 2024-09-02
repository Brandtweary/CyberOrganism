import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
from collections import deque
from training_statistics import TrainingStatistics
import copy
from typing import Dict, Any, Tuple, Callable, List, Optional
import numpy as np
from uuid import UUID, uuid4
from state_snapshot import StateSnapshot
from enum import Enum
from prioritized_experience_replay import PrioritizedExperienceReplay, Experience

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    NO_MOVE = 4

class DQNNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(DQNNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class DQN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(DQN, self).__init__()
        self.policy_net = DQNNetwork(input_size, hidden_size, output_size)
        self.target_net = DQNNetwork(input_size, hidden_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_net(x)

    def update_target_net(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

class Organism:
    def __init__(self, matrika, initial_position: Tuple[float, float]) -> None:
        self.id: UUID = uuid4()
        self.matrika = matrika
        self.input_parameters: List[str] = []
        self.display_parameters: List[str] = [
            'energy',
            'nutrition',
            'loss_avg',
            'reward_avg',
            'epsilon',
            'q_value_average',
            'expected_q_value_average'
        ]
        self.energy: float = 5.0
        self.nutrition: float = 0.0
        self.loss_avg: float = 0.0
        self.q_value_average: float = 0.0
        self.expected_q_value_average: float = 0.0
        self.epsilon: float = 1.0  # Start with 100% exploration
        self.epsilon_min: float = 0.01  # Minimum exploration rate
        self.epsilon_decay: float = 0.999  # Decay rate for epsilon
        self.boltzmann_temperature: float = 0.2
        self.movement_speed: float = 1.0
        self.attention_speed: float = 3.0 
        self.detection_radius: int = 100
        self.consumption_range: int = 3
        self.max_nearest_items: int = 1
        self.nearest_item_params: int = 3
        self.nearest_items: List[Any] = []
        self.nearest_item: Optional[Any] = None
        self.attention_move_distance: float = 1.0
        self.energy_consumption: float = 0.0
        self.nutrition_consumption: float = 0.0

        # Attention point attributes
        self.attention_x, self.attention_y = initial_position

        # DQN and training
        self.action_mapping = {action.value: action.name for action in Action}
        
        input_size: int = len(self.input_parameters) + self.nearest_item_params * self.max_nearest_items + 2
        hidden_size: int = 16
        output_size: int = len(self.action_mapping)
        self.dqn: DQN = DQN(input_size, hidden_size, output_size)
        self.optimizer: optim.Optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        self.gamma: float = 0.9
        self.training_steps: int = 0
        self.target_update: int = 10
        self.replay_buffer = PrioritizedExperienceReplay(capacity=100000, batch_size=64)
        self.batch_size: int = 32
        self.gradient_clip: float = 1.0
        

        # Training statistics and history
        self.training_stats: TrainingStatistics = TrainingStatistics(self.dqn, self.optimizer)
        self.loss_history: deque = deque(maxlen=100)
        self.q_value_history: deque = deque(maxlen=100)
        self.expected_q_history: deque = deque(maxlen=100)
        self.reward_history: deque = deque(maxlen=100)
        self.reward_avg: float = 0.0

        # Current experience
        self.current_experience = None
        self.current_reward = 0.0
        self.item_memory: List[UUID] = []  # New attribute to store remembered item IDs

    def clone(self: 'Organism', parent: 'Organism') -> None:
        """Clone the parent organism's DQN, optimizer, and training stats."""
        self.dqn = copy.deepcopy(parent.dqn)
        self.optimizer = type(parent.optimizer)(self.dqn.policy_net.parameters(), 
                                        lr=parent.optimizer.param_groups[0]['lr'])
        self.training_stats = TrainingStatistics(self.dqn, self.optimizer)

    def get_internal_state(self, external_state: StateSnapshot) -> Tuple[torch.Tensor, List[Optional[UUID]]]:
        organism_state = external_state['organisms'][str(self.id)]
        organism_x, organism_y = organism_state['x'], organism_state['y']
        
        state: List[float] = []
        nearest_item_ids: List[Optional[UUID]] = []

        # Add input parameters
        # NOTE: Do not delete this part. We might add more input parameters in the future.
        for param in self.input_parameters:
            state.append(getattr(self, param))

        nearest_item_ids = self.matrika.get_nearest_items(
            organism_x, organism_y, 
            self.max_nearest_items, 
            self.detection_radius,
            item_type='food',
            return_IDs=True
        )

        self.update_item_memory(nearest_item_ids)

        if not nearest_item_ids:
            nearest_item_ids = self.get_item_from_memory(organism_x, organism_y, external_state)

        # Process nearest items
        for item_id in nearest_item_ids:
            if str(item_id) in external_state['items']:
                item_state = external_state['items'][str(item_id)]
                
                distance_org_item = self.matrika.calculate_distance(organism_x, organism_y, item_state['x'], item_state['y'], normalize_to_viewport=True)
                direction_org_item = self.matrika.calculate_angle(organism_x, organism_y, item_state['x'], item_state['y'], normalize=True)
                
                state.extend([distance_org_item, direction_org_item, item_state['reward']])

        # Pad with zeros if fewer items than max_nearest_items
        padding_length = (self.max_nearest_items - len(nearest_item_ids)) * 3
        state.extend([0.0] * padding_length)

        # Add nearest item IDs padding
        nearest_item_ids.extend([None] * (self.max_nearest_items - len(nearest_item_ids)))

        # Add attention point info for the first (nearest) item, if it exists
        if nearest_item_ids and str(nearest_item_ids[0]) in external_state['items']:
            item_state = external_state['items'][str(nearest_item_ids[0])]
            distance_att_item = self.matrika.calculate_distance(self.attention_x, self.attention_y, item_state['x'], item_state['y'], normalize_to_viewport=True)
            direction_att_item = self.matrika.calculate_angle(self.attention_x, self.attention_y, item_state['x'], item_state['y'], normalize=True)
            state.extend([distance_att_item, direction_att_item])
        else:
            state.extend([0.0, 0.0])  # Pad with zeros if no nearest item

        # Add distance and direction from organism to attention point
        # distance_org_att = self.matrika.calculate_distance(organism_x, organism_y, self.attention_x, self.attention_y, normalize_to_viewport=True)
        # direction_org_att = self.matrika.calculate_angle(organism_x, organism_y, self.attention_x, self.attention_y, normalize=True)
        # state.extend([distance_org_att, direction_org_att])

        return torch.tensor(state, dtype=torch.float32), nearest_item_ids

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
            if self.matrika.item_exists(str(item_id)):
                item_state = external_state['items'][str(item_id)]
                distance = self.matrika.calculate_distance(organism_x, organism_y, item_state['x'], item_state['y'])
                if distance < nearest_distance:
                    nearest_item = item_id
                    nearest_distance = distance
            else:
                self.item_memory.remove(item_id)

        return [nearest_item] if nearest_item else []

    def select_attention_move(self, state: torch.Tensor) -> int:
        if random.random() < self.epsilon:
            # Epsilon-greedy exploration
            action_index = random.randint(0, len(self.action_mapping) - 1)
        else:
            with torch.no_grad():
                q_values = self.dqn.policy_net(state).squeeze()
            
            # Apply Boltzmann exploration
            action_index = self.boltzmann_exploration(q_values)
        
        return action_index

    def boltzmann_exploration(self, q_values: torch.Tensor) -> int:
        # Apply softmax with temperature
        probabilities = F.softmax(q_values / self.boltzmann_temperature, dim=0)
        
        # Convert to numpy for numpy's choice function
        probabilities = probabilities.cpu().numpy()
        
        # Choose action based on probabilities
        action_index = np.random.choice(len(self.action_mapping), p=probabilities)
        
        return action_index

    def update_attention_point(self, action_index: int) -> Tuple[float, float]:
        action = Action(action_index)
        dx, dy = 0, 0
        if action == Action.UP:
            dy = -self.attention_move_distance
        elif action == Action.DOWN:
            dy = self.attention_move_distance
        elif action == Action.LEFT:
            dx = -self.attention_move_distance
        elif action == Action.RIGHT:
            dx = self.attention_move_distance
        elif action == Action.NO_MOVE:
            return 0, 0

        return dx * self.attention_speed, dy * self.attention_speed

    def move(self, external_state: StateSnapshot, attention_vector: Tuple[float, float]) -> Tuple[float, float]:
        org_state = external_state['organisms'][str(self.id)]
        org_x, org_y = org_state['x'], org_state['y']
        current_attention_x, current_attention_y = org_state['attention_point']
        
        # Calculate theoretical new attention point
        theoretical_attention_x = current_attention_x + attention_vector[0]
        theoretical_attention_y = current_attention_y + attention_vector[1]
        
        dx = theoretical_attention_x - org_x
        dy = theoretical_attention_y - org_y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > 0:
            # Normalize the direction vector
            direction_x = dx / distance
            direction_y = dy / distance
            
            # Create movement vector with magnitude equal to movement_speed
            move_x = direction_x * self.movement_speed
            move_y = direction_y * self.movement_speed
            
            return move_x, move_y
        else:
            return 0, 0

    def update_state(self, external_state: StateSnapshot) -> Dict[str, Any]:
        """
        Update the organism's state based on the current external state.
        
        This method should be a clean sequence of function calls, with minimal
        non-method code. It orchestrates the state update process by calling
        other methods to perform specific tasks.
        
        Args:
            external_state (StateSnapshot): The current external state of the environment.
        
        Returns:
            Dict[str, Any]: A dictionary containing the changes to be applied to the external state.
        """
        internal_state, nearest_item_ids = self.get_internal_state(external_state)
        action_index = self.select_attention_move(internal_state)
        attention_vector = self.update_attention_point(action_index)
        move_x, move_y = self.move(external_state, attention_vector)
        
        external_state_change = {
            'movement_vector': (move_x, move_y),
            'attention_vector': attention_vector,
            'nearest_item_id': nearest_item_ids[0] if nearest_item_ids else None
        }
        
        metabolism_changes = self.update_metabolism()
        
        external_state_change.update(metabolism_changes)
        
        # Store the current experience
        self.current_experience = {
            'state': internal_state,
            'action': action_index,
            'reward': None,  # Will be set in apply_state
            'next_state': None  # Will be set in apply_state
        }
        
        return external_state_change

    def calculate_rewards(self, old_state: StateSnapshot, new_state: StateSnapshot) -> float:
        old_organism_state = old_state['organisms'][str(self.id)]
        new_organism_state = new_state['organisms'][str(self.id)]
        
        old_attention_x, old_attention_y = old_organism_state['attention_point']
        new_attention_x, new_attention_y = new_organism_state['attention_point']
        
        nearest_item_id = new_organism_state.get('nearest_item_id')
        
        if nearest_item_id and nearest_item_id in new_state['items']:
            item_state = new_state['items'][str(nearest_item_id)]
            item_x, item_y = item_state['x'], item_state['y']
            
            new_distance = self.matrika.calculate_distance(new_attention_x, new_attention_y, item_x, item_y)
            
            if new_distance > 2:
                # Calculate proximity penalty
                normalized_distance = self.matrika.calculate_distance(new_attention_x, new_attention_y, item_x, item_y, normalize_to_viewport=True)
                proximity_penalty = self.proximity_penalty(normalized_distance, method='exponential')
                
                # Calculate direction reward
                attention_delta_x = new_attention_x - old_attention_x
                attention_delta_y = new_attention_y - old_attention_y

                if attention_delta_x == 0 and attention_delta_y == 0:
                    movement_angle = 0  # No movement
                else:
                    movement_angle = math.atan2(attention_delta_y, attention_delta_x)

                target_delta_x = item_x - old_attention_x
                target_delta_y = item_y - old_attention_y
                target_angle = math.atan2(target_delta_y, target_delta_x)

                angle_diff = math.atan2(math.sin(target_angle - movement_angle), math.cos(target_angle - movement_angle))

                direction_reward = self.calculate_direction_reward(angle_diff)
                
                # Combine rewards
                reward = proximity_penalty + direction_reward
            else:
                # Focus reward when close to the target
                reward = 1.0  # Base reward for being close
                
                attention_movement = self.matrika.calculate_distance(old_attention_x, old_attention_y, new_attention_x, new_attention_y)
                if attention_movement < 0.1:
                    reward += 1.0  # Additional reward for staying still when close
        else:
            reward = -0.444  # Penalty if there's no nearest item
        
        # Add the current_reward from food consumption        reward += self.current_reward
        self.current_reward = 0.0  # Reset the current_reward after incorporating it
        
        # Update reward history and average
        self.reward_history.append(reward)
        self.reward_avg = sum(self.reward_history) / len(self.reward_history)
        
        return reward

    def calculate_direction_reward(self, angle_diff: float) -> float:
        # Convert angle difference to degrees and take the absolute value
        angle_diff_degrees = abs(math.degrees(angle_diff))
        
        # Define the range for positive rewards (±45 degrees)
        positive_range = 45
        
        if angle_diff_degrees <= positive_range:
            # Positive reward for angles within the correct range
            # Scaled so that 0 degrees gives 1.0 and 45 degrees gives 0.0
            return 1.0 - (angle_diff_degrees / positive_range)
        else:
            # Negative reward for angles outside the correct range
            # Scaled so that 45 degrees gives 0.0 and 180 degrees gives -1.0
            return -1.0 * (angle_diff_degrees - positive_range) / (180 - positive_range)


    def proximity_penalty(self, normalized_distance, max_penalty=0.5, transition_start=0.2, transition_end=0.1, method='quadratic'):
        def smooth_step(x):
            # Smooth step function
            x = max(0, min(1, (x - transition_end) / (transition_start - transition_end)))
            return x * x * (3 - 2 * x)
        
        if method == 'quadratic':
            # Quadratic penalty
            penalty = max_penalty * (normalized_distance ** 2)
        elif method == 'exponential':
            # Exponential penalty
            penalty = max_penalty * (math.exp(normalized_distance) - 1) / (math.e - 1)
        else:
            raise ValueError("Invalid method. Choose 'quadratic' or 'exponential'.")
        
        # Smooth step component
        step_factor = smooth_step(normalized_distance)
        
        # Combine penalty and step components
        return -penalty * step_factor


    def apply_state(self, old_state: StateSnapshot, new_state: StateSnapshot) -> None:
        total_reward = self.calculate_rewards(old_state, new_state)
        new_internal_state, _ = self.get_internal_state(new_state)
        
        experience = Experience(
            state=self.current_experience['state'],
            action=self.current_experience['action'],
            reward=total_reward,
            next_state=new_internal_state
        )
        
        self.replay_buffer.add(experience)
        
        self.current_experience = None
        
        self.train()

    def update_metabolism(self) -> Dict[str, Any]:
        self.energy -= self.energy_consumption
        self.nutrition = max(0, self.nutrition - self.nutrition_consumption)
        
        should_spawn = self.handle_reproduction()
        
        return {
            'alive': self.energy > 0,
            'spawn': should_spawn
        }

    def handle_reproduction(self) -> bool:
        if self.energy > 100 and self.nutrition > 100:
            self.energy -= 50
            self.nutrition -= 30
            return True
        return False

    def train(self) -> None:
        if self.replay_buffer.tree.n_entries < self.batch_size:
            return

        batch, idxs, weights = self.replay_buffer.sample()
        states, actions, rewards, next_states = zip(*batch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        current_q_values = self.dqn.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.dqn.target_net(next_states).max(1)[0]
        
        expected_q_values = rewards + (self.gamma * next_q_values)

        # Use Huber loss with importance sampling weights
        losses = F.huber_loss(current_q_values, expected_q_values.unsqueeze(1), reduction='none', delta=1.0)
        loss = (losses * weights.unsqueeze(1)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.policy_net.parameters(), max_norm=self.gradient_clip)
        self.optimizer.step()

        # Update priorities
        new_priorities = losses.detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(idxs, new_priorities)

        self.loss_history.append(loss.item())
        self.q_value_history.append(current_q_values.mean().item())
        self.expected_q_history.append(expected_q_values.mean().item())

        self.loss_avg = sum(self.loss_history) / len(self.loss_history)
        self.q_value_average = sum(self.q_value_history) / len(self.q_value_history)
        self.expected_q_value_average = sum(self.expected_q_history) / len(self.expected_q_history)

        self.training_steps += 1
        if self.training_steps % self.target_update == 0:
            self.dqn.update_target_net()
    
        self.decay_epsilon()  # Decay epsilon after each training step

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)