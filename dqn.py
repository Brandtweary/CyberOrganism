import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from typing import Any, Dict, List
from enum import Action
from RL_neural_network import ReinforcementLearningNeuralNetwork


class DQNNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, hidden_layers: int):
        super(DQNNetwork, self).__init__()
        layers: List[nn.Module] = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class DQN(ReinforcementLearningNeuralNetwork):
    def __init__(self, organism: Any, action_mapping: Dict[int, str], input_size: int, hidden_size: int, output_size: int, hidden_layers: int, learning_rate: float):
        super().__init__(organism, action_mapping, input_size, hidden_size, output_size, hidden_layers, learning_rate)
        self.target_network = self.create_network()
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()

    def create_network(self) -> nn.Module:
        return DQNNetwork(self.input_size, self.hidden_size, self.output_size, self.hidden_layers)

    def create_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(self.main_network.parameters(), lr=self.learning_rate)

    def select_action(self, state: torch.Tensor) -> Action:
        if random.random() < self.organism.epsilon:
            action_index = random.randint(0, self.output_size - 1)
        else:
            with torch.no_grad():
                q_values = self.main_network(state).squeeze()
            action_index = self.boltzmann_exploration(q_values)
        
        return Action(self.action_mapping[action_index])

    def boltzmann_exploration(self, q_values: torch.Tensor) -> int:
        probabilities = F.softmax(q_values / self.organism.boltzmann_temperature, dim=0)
        probabilities = probabilities.cpu().numpy()
        action_index = np.random.choice(self.output_size, p=probabilities)
        return action_index

    def learn(self) -> Dict[str, float]:
        if not self.organism.replay_buffer.can_sample():
            return {"avg_loss": 0.0, "avg_q_value": 0.0, "avg_expected_q_value": 0.0}

        batch, idxs = self.organism.replay_buffer.sample()
        
        self.optimizer.zero_grad()
        total_loss = 0
        total_q_value = 0
        total_expected_q_value = 0
        
        for i, experience in enumerate(batch):
            state, action, reward, next_state = experience
            
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            
            # Convert Action enum to integer index using the inverse of action_mapping
            action_index = next(index for index, action_name in self.action_mapping.items() if action_name == action.name)
            action_index = torch.tensor([action_index], dtype=torch.long)
            
            reward = torch.tensor([reward], dtype=torch.float32)

            current_q_value = self.main_network(state).gather(1, action_index.unsqueeze(1))

            with torch.no_grad():
                next_q_value = self.target_network(next_state).max(1)[0].detach()

            expected_q_value = reward + (self.organism.gamma * next_q_value)

            loss = F.smooth_l1_loss(current_q_value, expected_q_value.unsqueeze(1))
            
            loss.backward()
            
            total_loss += loss.item()
            total_q_value += current_q_value.item()
            total_expected_q_value += expected_q_value.item()

            td_error = abs(current_q_value.item() - expected_q_value.item())
            self.organism.replay_buffer.update_priorities([idxs[i]], [td_error])

        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=self.organism.gradient_clip)
        
        self.optimizer.step()

        avg_loss = total_loss / len(batch)
        avg_q_value = total_q_value / len(batch)
        avg_expected_q_value = total_expected_q_value / len(batch)

        self.organism.training_steps += 1
        if self.organism.training_steps % self.organism.target_update == 0:
            self.update_target_network()

        self.decay_epsilon()

        return {
            "avg_loss": avg_loss,
            "avg_q_value": avg_q_value,
            "avg_expected_q_value": avg_expected_q_value
        }

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.main_network.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)