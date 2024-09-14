import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from typing import Any, Dict, List
from enums import Action
from RL_algorithm import ReinforcementLearningAlgorithm


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

class DQN(ReinforcementLearningAlgorithm):
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
        
        return self.action_mapping[action_index]

    def boltzmann_exploration(self, q_values: torch.Tensor) -> int:
        probabilities = F.softmax(q_values / self.organism.boltzmann_temperature, dim=0)
        probabilities = probabilities.cpu().numpy()
        action_index = np.random.choice(self.output_size, p=probabilities)
        return action_index

    def learn(self) -> Dict[str, Any]:
        if not self.organism.replay_buffer.can_sample():
            return {}

        batch, idxs = self.organism.replay_buffer.sample()
        
        self.optimizer.zero_grad()
        metrics = {action_value: {"loss": [], "current_q": [], "expected_q": []} for action_value in self.action_mapping.keys()}
        
        for i, experience in enumerate(batch):
            state, action, reward, next_state = experience
            
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action_index = torch.tensor([action.value], dtype=torch.long) 
            reward = torch.tensor([reward], dtype=torch.float32)
            current_q_value = self.main_network(state).gather(1, action_index.unsqueeze(1))

            with torch.no_grad():
                next_q_value = self.target_network(next_state).max(1)[0].detach()

            expected_q_value = reward + (self.organism.gamma * next_q_value)

            loss = F.smooth_l1_loss(current_q_value, expected_q_value.unsqueeze(1))
            
            loss.backward()
            
            metrics[action.value]["loss"].append(loss.item())
            metrics[action.value]["current_q"].append(current_q_value.item())
            metrics[action.value]["expected_q"].append(expected_q_value.item())

            td_error = abs(current_q_value.item() - expected_q_value.item())
            self.organism.replay_buffer.update_priorities([idxs[i]], [td_error])

        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=self.organism.gradient_clip)
        
        self.optimizer.step()

        self.organism.training_steps += 1
        if self.organism.training_steps % self.organism.target_update == 0:
            self.update_target_network()

        self.decay_epsilon()

        return metrics

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.main_network.state_dict())
    
    def decay_epsilon(self):
        self.organism.epsilon = max(self.organism.epsilon_min, self.organism.epsilon * self.organism.epsilon_decay)