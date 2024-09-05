import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
from collections import deque
from training_statistics import TrainingStatistics
import copy
from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
from uuid import UUID, uuid4
from state_snapshot import StateSnapshot, ObjectType
from prioritized_experience_replay import PrioritizedExperienceReplay, Experience
from enums import Action
from shared_resources import calculate_synchronized_params


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
    def __init__(self, SimulationEngine: Any, initial_position: Tuple[int, int]) -> None:
        self.id: UUID = uuid4()
        self.sim_engine = SimulationEngine
        self.input_parameters: List[str] = [
            'attention_item_distance',
            'attention_item_direction',
            'organism_attention_distance',
            'organism_attention_direction'
        ]
        self.display_parameters: List[str] = [
            'energy',
            'nutrition',
            'loss_avg',
            'reward_avg',
            'epsilon',
            'q_value_average',
            'expected_q_value_average'
        ]
        self.x: float = initial_position[0]
        self.y: float = initial_position[1]
        self.attention_x: float = initial_position[0]
        self.attention_y: float = initial_position[1]
        self.marked_for_deletion: bool = False
        self.type: ObjectType = ObjectType.ORGANISM
        self.collision: bool = True
        self.consumable: bool = False
        self.energy: float = 5.0
        self.nutrition: float = 0.0
        self.loss_avg: float = 0.0
        self.q_value_average: float = 0.0
        self.expected_q_value_average: float = 0.0
        self.epsilon: float = 1.0
        self.epsilon_min: float = 0.01
        self.epsilon_decay: float = 0.999
        self.boltzmann_temperature: float = 0.2
        self.movement_speed: float = 1.0
        self.attention_speed: float = 3.0 
        self.detection_radius: int = 100
        self.consumption_range: int = 3
        self.max_nearest_items: int = 1
        self.nearest_item_params: int = 3
        self.nearest_items: List[Any] = []
        self.nearest_item_ID: Optional[UUID] = None
        self.attention_move_distance: float = 1.0
        self.energy_consumption: float = 0.0
        self.nutrition_consumption: float = 0.0

        self.action_mapping: Dict[int, str] = {action.value: action.name for action in Action}
        
        input_size: int = len(self.input_parameters) + self.max_nearest_items * self.nearest_item_params
        hidden_size: int = 16
        output_size: int = len(self.action_mapping)
        self.dqn: DQN = DQN(input_size, hidden_size, output_size)
        self.optimizer: optim.Optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        self.gamma: float = 0.9
        self.training_steps: int = 0
        self.target_update: int = 100
        self.batch_size: int = 4
        self.capacity: int = 100000
        self.replay_buffer = PrioritizedExperienceReplay(capacity=self.capacity, batch_size=self.batch_size)
        self.gradient_clip: float = 1.0

        self.training_stats: TrainingStatistics = TrainingStatistics(self.dqn, self.optimizer)
        self.loss_history: deque = deque(maxlen=100)
        self.q_value_history: deque = deque(maxlen=100)
        self.expected_q_history: deque = deque(maxlen=100)
        self.reward_history: deque = deque(maxlen=100)
        self.reward_avg: float = 0.0

        self.current_experience: Optional[Dict[str, Any]] = None
        self.current_reward: float = 0.0
        self.item_memory: List[UUID] = []

        self.synchronized_params: List[str] = calculate_synchronized_params(self)

    def clone(self: 'Organism', parent: 'Organism') -> None:
        self.dqn = copy.deepcopy(parent.dqn)
        self.optimizer = type(parent.optimizer)(self.dqn.policy_net.parameters(), 
                                        lr=parent.optimizer.param_groups[0]['lr'])
        self.training_stats = TrainingStatistics(self.dqn, self.optimizer)

    def get_internal_state(self, external_state: StateSnapshot) -> torch.Tensor:
        nearest_items_vector = self.calculate_nearest_items_vector(external_state)
        self.calculate_attention_item_parameters(external_state)
        self.calculate_organism_attention_parameters()

        state = [getattr(self, param) for param in self.input_parameters]
        state.extend(nearest_items_vector)

        return torch.tensor(state, dtype=torch.float32)

    def calculate_nearest_items_vector(self, external_state: StateSnapshot) -> List[float]:
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

    def select_attention_move(self, state: torch.Tensor) -> int:
        if random.random() < self.epsilon:
            action_index = random.randint(0, len(self.action_mapping) - 1)
        else:
            with torch.no_grad():
                q_values = self.dqn.policy_net(state).squeeze()
            
            action_index = self.boltzmann_exploration(q_values)
        
        return action_index

    def boltzmann_exploration(self, q_values: torch.Tensor) -> int:
        probabilities = F.softmax(q_values / self.boltzmann_temperature, dim=0)
        probabilities = probabilities.cpu().numpy()
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

    def move(self, attention_vector: Tuple[float, float]) -> Tuple[float, float]:
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
        internal_state = self.get_internal_state(external_state)
        action_index = self.select_attention_move(internal_state)
        attention_vector = self.update_attention_point(action_index)
        move_x, move_y = self.move(attention_vector)
        
        external_state_change = {
            'movement_vector': (move_x, move_y),
            'attention_vector': attention_vector,
        }
        
        metabolism_changes = self.update_metabolism()
        
        external_state_change.update(metabolism_changes)
        
        self.current_experience = {
            'state': internal_state,
            'action': action_index,
            'reward': None,  # Will be set in apply_state
            'next_state': None  # Will be set in apply_state
        }
        
        return external_state_change

    def calculate_rewards(self, old_state: StateSnapshot, new_state: StateSnapshot) -> float:
        old_organism_state = old_state.get_state(self.id)
        
        old_attention_x, old_attention_y = old_organism_state['attention_x'], old_organism_state['attention_y']
        new_attention_x, new_attention_y = self.attention_x, self.attention_y
        
        nearest_item_id = self.nearest_item_ID
        
        if nearest_item_id and nearest_item_id in new_state['items']:
            item_state = new_state['items'][str(nearest_item_id)]
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
            reward = -0.666
        
        reward += self.current_reward
        self.current_reward = 0.0
        
        self.reward_history.append(reward)
        self.reward_avg = sum(self.reward_history) / len(self.reward_history)
        
        return reward
    
    def calculate_direction_reward(self, angle_diff: float) -> float:
        normalized_angle = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        reward = 0.5 * math.cos(normalized_angle)
        return reward

    def apply_state(self, old_state: StateSnapshot, new_state: StateSnapshot) -> None:
        total_reward = self.calculate_rewards(old_state, new_state)
        new_internal_state = self.get_internal_state(new_state)
        
        experience = Experience(
            state=self.current_experience['state'],
            action=self.current_experience['action'],
            reward=total_reward,
            next_state=new_internal_state
        )
        
        self.replay_buffer.add(experience)
        
        self.current_experience = None
        
        self.learn()

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

    def learn(self) -> None:
        if not self.replay_buffer.can_sample():
            return

        batch, idxs = self.replay_buffer.sample()
        
        self.optimizer.zero_grad()
        total_loss = 0
        total_q_value = 0
        total_expected_q_value = 0
        
        for i, experience in enumerate(batch):
            state, action, reward, next_state = experience
            
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = torch.tensor([action], dtype=torch.long)
            reward = torch.tensor([reward], dtype=torch.float32)

            current_q_values = self.dqn.policy_net(state).gather(1, action.unsqueeze(1))

            with torch.no_grad():
                next_q_values = self.dqn.target_net(next_state).max(1)[0].detach()

            expected_q_values = reward + (self.gamma * next_q_values)

            loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
            
            loss.backward()
            
            total_loss += loss.item()
            total_q_value += current_q_values.item()
            total_expected_q_value += expected_q_values.item()

            td_error = abs(current_q_values.item() - expected_q_values.item())
            self.replay_buffer.update_priorities([idxs[i]], [td_error])

        torch.nn.utils.clip_grad_norm_(self.dqn.policy_net.parameters(), max_norm=self.gradient_clip)
        
        self.optimizer.step()

        self.loss_history.append(total_loss / len(batch))
        self.q_value_history.append(total_q_value / len(batch))
        self.expected_q_history.append(total_expected_q_value / len(batch))
        
        self.loss_avg = sum(self.loss_history) / len(self.loss_history)
        self.q_value_average = sum(self.q_value_history) / len(self.q_value_history)
        self.expected_q_value_average = sum(self.expected_q_history) / len(self.expected_q_history)

        self.training_steps += 1
        if self.training_steps % self.target_update == 0:
            self.dqn.update_target_net()

        self.decay_epsilon()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)