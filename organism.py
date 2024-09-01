import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque
from training_statistics import TrainingStatistics
import copy
from typing import Dict, Any, Tuple, Callable, List, Optional
import numpy as np
from uuid import UUID, uuid4
from state_view import StateView

class DQNNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(DQNNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
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
        self.display_parameters: List[str] = ['energy', 'nutrition', 'loss_avg', 'reward_avg', 'epsilon']
        self.energy: float = 5.0
        self.nutrition: float = 0.0
        self.loss_avg: float = 0.0
        self.epsilon: float = 1.0  # Start with 100% exploration
        self.epsilon_min: float = 0.01  # Minimum exploration rate
        self.epsilon_decay: float = 0.995  # Decay rate for epsilon
        self.movement_speed: float = 1.0
        self.attention_speed: float = 5.0  # 5x faster than movement_speed
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
        input_size: int = len(self.input_parameters) + self.nearest_item_params * self.max_nearest_items + 4
        hidden_size: int = 16
        output_size: int = 4  # 4 discrete actions: up, down, left, right
        self.dqn: DQN = DQN(input_size, hidden_size, output_size)
        self.optimizer: optim.Optimizer = optim.Adam(self.dqn.parameters(), lr=1e-3)
        self.gamma: float = 0.9
        self.training_steps: int = 0
        self.target_update: int = 10
        self.experience_buffer: deque = deque(maxlen=10000)

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

    def clone(self: 'Organism', parent: 'Organism') -> None:
        """Clone the parent organism's DQN, optimizer, and training stats."""
        self.dqn = copy.deepcopy(parent.dqn)
        self.optimizer = type(parent.optimizer)(self.dqn.policy_net.parameters(), 
                                        lr=parent.optimizer.param_groups[0]['lr'])
        self.training_stats = TrainingStatistics(self.dqn, self.optimizer)

    def get_internal_state(self, external_state: StateView) -> Tuple[torch.Tensor, List[Optional[UUID]]]:
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

        # Process nearest items
        for item_id in nearest_item_ids:
            if str(item_id) in external_state['items']:
                item_state = external_state['items'][str(item_id)]
                
                distance_org_item = self.matrika.calculate_distance(organism_x, organism_y, item_state['x'], item_state['y'])
                direction_org_item = self.matrika.calculate_angle(organism_x, organism_y, item_state['x'], item_state['y'])
                
                state.extend([distance_org_item, direction_org_item, item_state['reward']])

        # Pad with zeros if fewer items than max_nearest_items
        padding_length = (self.max_nearest_items - len(nearest_item_ids)) * 3
        state.extend([0.0] * padding_length)

        # Add nearest item IDs padding
        nearest_item_ids.extend([None] * (self.max_nearest_items - len(nearest_item_ids)))

        # Add attention point info for the first (nearest) item, if it exists
        if nearest_item_ids and str(nearest_item_ids[0]) in external_state['items']:
            item_state = external_state['items'][str(nearest_item_ids[0])]
            distance_att_item = self.matrika.calculate_distance(self.attention_x, self.attention_y, item_state['x'], item_state['y'])
            direction_att_item = self.matrika.calculate_angle(self.attention_x, self.attention_y, item_state['x'], item_state['y'])
            state.extend([distance_att_item, direction_att_item])
        else:
            state.extend([0.0, 0.0])  # Pad with zeros if no nearest item

        # Add distance and direction from organism to attention point
        distance_org_att = self.matrika.calculate_distance(organism_x, organism_y, self.attention_x, self.attention_y)
        direction_org_att = self.matrika.calculate_angle(organism_x, organism_y, self.attention_x, self.attention_y)
        state.extend([distance_org_att, direction_org_att])

        return torch.tensor(state, dtype=torch.float32), nearest_item_ids

    def select_attention_move(self, state: torch.Tensor) -> int:
        if random.random() < self.epsilon:
            action = random.randint(0, 3)  # Explore: choose a random action
        else:
            with torch.no_grad():
                q_values = self.dqn.policy_net(state)
                action = q_values.argmax().item()  # Exploit: choose the best action

        return action

    def update_attention_point(self, action: int) -> Tuple[float, float]:
        dx, dy = 0, 0
        if action == 0:  # Move up
            dy = -self.attention_move_distance
        elif action == 1:  # Move down
            dy = self.attention_move_distance
        elif action == 2:  # Move left
            dx = -self.attention_move_distance
        elif action == 3:  # Move right
            dx = self.attention_move_distance

        return dx * self.attention_speed, dy * self.attention_speed

    def move(self, external_state: StateView, attention_vector: Tuple[float, float]) -> Tuple[float, float]:
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

    def update_state(self, external_state: StateView) -> Dict[str, Any]:
        """
        Update the organism's state based on the current external state.
        
        This method should be a clean sequence of function calls, with minimal
        non-method code. It orchestrates the state update process by calling
        other methods to perform specific tasks.
        
        Args:
            external_state (StateView): The current external state of the environment.
        
        Returns:
            Dict[str, Any]: A dictionary containing the changes to be applied to the external state.
        """
        internal_state, nearest_item_ids = self.get_internal_state(external_state)
        action = self.select_attention_move(internal_state)
        attention_vector = self.update_attention_point(action)
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
            'action': action,
            'reward': None,  # Will be set in apply_state
            'next_state': None  # Will be set in apply_state
        }
        
        return external_state_change

    def calculate_rewards(self, old_state: StateView, new_state: StateView) -> float:
        old_organism_state = old_state['organisms'][str(self.id)]
        new_organism_state = new_state['organisms'][str(self.id)]
        
        old_attention_x, old_attention_y = old_organism_state['attention_point']
        new_attention_x, new_attention_y = new_organism_state['attention_point']
        
        nearest_item_id = new_organism_state.get('nearest_item_id')
        
        if nearest_item_id and nearest_item_id in new_state['items']:
            item_state = new_state['items'][str(nearest_item_id)]
            
            old_distance = self.matrika.calculate_distance(old_attention_x, old_attention_y, item_state['x'], item_state['y'])
            new_distance = self.matrika.calculate_distance(new_attention_x, new_attention_y, item_state['x'], item_state['y'])
            
            distance_improvement = old_distance - new_distance
            
            # Reward based on distance improvement (positive or negative)
            reward = 0.5 * math.copysign(math.log1p(abs(distance_improvement) * 100), distance_improvement)
        else:
            reward = 0  # No reward if there's no nearest item
        
        # Add the current_reward from food consumption
        reward += self.current_reward
        self.current_reward = 0.0  # Reset the current_reward after incorporating it
        
        # Update reward history and average
        self.reward_history.append(reward)
        self.reward_avg = sum(self.reward_history) / len(self.reward_history)
        
        return reward

    def apply_state(self, old_state: StateView, new_state: StateView) -> None:
        total_reward = self.calculate_rewards(old_state, new_state)
        new_internal_state, _ = self.get_internal_state(new_state)  # Unpack the tuple
        
        self.current_experience.update({
            'reward': total_reward,
            'next_state': new_internal_state  # Store only the tensor
        })
        
        self.experience_buffer.append(tuple(self.current_experience.values()))
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
        if len(self.experience_buffer) < 16:
            return

        batch = random.sample(self.experience_buffer, 16)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        current_q_values = self.dqn.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.dqn.target_net(next_states).max(1)[0]
        
        expected_q_values = rewards + (self.gamma * next_q_values)

        loss = nn.functional.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.loss_history.append(loss.item())
        self.q_value_history.append(current_q_values.mean().item())
        self.expected_q_history.append(expected_q_values.mean().item())

        self.loss_avg = sum(self.loss_history) / len(self.loss_history)

        self.training_steps += 1
        if self.training_steps % self.target_update == 0:
            self.dqn.update_target_net()
    
        self.decay_epsilon()  # Decay epsilon after each training step

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)