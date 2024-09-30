import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from typing import Any, Dict, List
from enums import Action
from RL_algorithm import ReinforcementLearningAlgorithm
import threading
import copy
from state_snapshot import StateSnapshot
import queue

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
        self.target_network = self.create_network().to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()
        self.target_network_buffer = [self.target_network, copy.deepcopy(self.target_network)]
        self.target_buffer = 0

    def create_network(self) -> nn.Module:
        return DQNNetwork(self.input_size, self.hidden_size, self.output_size, self.hidden_layers)

    def create_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(self.main_network.parameters(), lr=self.learning_rate)

    def select_action(self, state: torch.Tensor) -> int:
        state = state.to(self.device)
        with self.network_lock:
            if random.random() < self.organism.epsilon:
                self.decay_epsilon()
                return random.randint(0, self.output_size - 1)
            else:
                with torch.no_grad():
                    q_values = self.main_network_buffer[self.current_buffer](state)
                    self.decay_epsilon()
                    return self.boltzmann_exploration(q_values)
    
    def decay_epsilon(self):
        self.organism.epsilon = max(self.organism.epsilon_min, self.organism.epsilon * self.organism.epsilon_decay)

    def boltzmann_exploration(self, q_values: torch.Tensor) -> int:
        probabilities = F.softmax(q_values / self.organism.boltzmann_temperature, dim=0)
        action_index = torch.multinomial(probabilities, 1).item()
        return action_index

    def _learn(self, state_snapshot: StateSnapshot) -> Dict[str, Any]:
        if not self.organism.replay_buffer.can_sample():
            return {}

        # Get all necessary parameters from state snapshot
        organism_state = state_snapshot.get_state(self.organism.id)
        gamma = organism_state['gamma']
        gradient_clip = organism_state['gradient_clip']
        target_update = organism_state['target_update']
        
        batch, idxs = self.organism.replay_buffer.sample()
        
        learning_network = self.main_network_buffer[1 - self.current_buffer]
        target_network = self.target_network_buffer[1 - self.target_buffer]

        self.optimizer.zero_grad()
        
        # Prepare batch data
        states, actions, rewards, next_states = zip(*batch)
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)

        # Compute Q-values for current states
        q_values = learning_network(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Compute Q-values for next states
            next_q_values = target_network(next_states)
            # Get maximum Q-value for next states
            max_next_q_values = next_q_values.max(1)[0]

        # Compute expected Q-values
        expected_q_values = rewards + (gamma * max_next_q_values)

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(learning_network.parameters(), max_norm=gradient_clip)
        
        self.optimizer.step()

        # Update priorities in replay buffer
        td_errors = abs(current_q_values.detach() - expected_q_values).tolist()
        self.organism.replay_buffer.update_priorities(idxs, td_errors)

        with self.network_lock:
            self.main_network_buffer[self.current_buffer].load_state_dict(learning_network.state_dict())
            self.current_buffer = 1 - self.current_buffer
            self.training_steps += 1  # remember to sync this differently if you use concurrent learn threads
            if self.training_steps % target_update == 0:
                self.target_network_buffer[self.target_buffer].load_state_dict(learning_network.state_dict())
                self.target_buffer = 1 - self.target_buffer

        return {
            "average_loss": loss.item(),
            "average_q_value": current_q_values.mean().item()
        }
