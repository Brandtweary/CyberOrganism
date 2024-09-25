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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.main_network = self.main_network.to(self.device)  # should probably be moved to base class
        self.target_network = self.create_network().to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()
        self.network_lock = threading.Lock()  # this should be moved to base class
        self.main_network_buffer = [self.main_network, copy.deepcopy(self.main_network)]
        self.target_network_buffer = [self.target_network, copy.deepcopy(self.target_network)]
        self.current_buffer = 0
        self.epsilon_buffer = [organism.epsilon, organism.epsilon]  # Initialize both with the same epsilon
        self.learn_queue = queue.Queue()
        self.learn_thread = threading.Thread(target=self._learn_worker, daemon=True)
        self.learn_thread.start()

    def create_network(self) -> nn.Module:
        return DQNNetwork(self.input_size, self.hidden_size, self.output_size, self.hidden_layers)

    def create_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(self.main_network.parameters(), lr=self.learning_rate)

    def select_action(self, state: torch.Tensor) -> int:
        state = state.to(self.device)
        if random.random() < self.epsilon_buffer[self.current_buffer]:
            return random.randint(0, self.output_size - 1)
        else:
            with torch.no_grad():
                q_values = self.main_network_buffer[self.current_buffer](state)
                return q_values.argmax().item()

    def boltzmann_exploration(self, q_values: torch.Tensor) -> int:
        probabilities = F.softmax(q_values / self.organism.boltzmann_temperature, dim=0)
        action_index = torch.multinomial(probabilities, 1).item()
        return action_index

    def _learn(self, state_snapshot: StateSnapshot) -> Dict[str, Any]:
        if not self.organism.replay_buffer.can_sample():
            return {}

        # Get hyperparameters from state
        organism_state = state_snapshot.get_state(self.organism.id)
        learning_rate = organism_state['learning_rate']
        gamma = organism_state['gamma']
        gradient_clip = organism_state['gradient_clip']
        epsilon = self.epsilon_buffer[self.current_buffer]
        epsilon_min = organism_state['epsilon_min']
        epsilon_decay = organism_state['epsilon_decay']
        
        batch, idxs = self.organism.replay_buffer.sample()
        
        # Use the non-current buffer for both main and target networks
        learning_network = self.main_network_buffer[1 - self.current_buffer]
        target_network = self.target_network_buffer[1 - self.current_buffer]
        
        temp_optimizer = optim.AdamW(learning_network.parameters(), lr=learning_rate)
        
        temp_optimizer.zero_grad()
        metrics = {self.action_mapping[i].name: {"loss": [], "current_q": [], "expected_q": []} for i in range(self.output_size)}
        
        for i, experience in enumerate(batch):
            state, action, reward, next_state = experience
            
            # Ensure tensors are on the correct device
            state = state.to(self.device)
            next_state = next_state.to(self.device)
            reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

            # Compute Q-values for current state
            q_values = learning_network(state)
            
            # Get Q-value for the taken action
            current_q_value = q_values[action]

            with torch.no_grad():
                # Compute Q-values for next state
                next_q_values = target_network(next_state)
                # Get maximum Q-value for next state
                next_q_value = next_q_values.max()

            # Compute expected Q-value
            expected_q_value = reward + (gamma * next_q_value)

            # Compute loss
            loss = F.smooth_l1_loss(current_q_value.unsqueeze(0), expected_q_value.unsqueeze(0))
            
            loss.backward()
            
            metrics[self.action_mapping[action].name]["loss"].append(loss.item())
            metrics[self.action_mapping[action].name]["current_q"].append(current_q_value.item())
            metrics[self.action_mapping[action].name]["expected_q"].append(expected_q_value.item())

            td_error = abs(current_q_value.item() - expected_q_value.item())
            self.organism.replay_buffer.update_priorities([idxs[i]], [td_error])

        torch.nn.utils.clip_grad_norm_(learning_network.parameters(), max_norm=gradient_clip)
        
        temp_optimizer.step()

        with self.network_lock:
            # Swap buffers
            next_buffer = 1 - self.current_buffer
            self.current_buffer = next_buffer
            # Update the main network buffer with the learned weights
            self.main_network_buffer[self.current_buffer].load_state_dict(learning_network.state_dict())
            # Update epsilon for the next buffer
            new_epsilon = max(epsilon_min, epsilon * epsilon_decay)
            self.epsilon_buffer[next_buffer] = new_epsilon
            # Update target network if necessary
            self.organism.training_steps += 1
            if self.organism.training_steps % self.organism.target_update == 0:
                self.target_network_buffer[self.current_buffer].load_state_dict(self.main_network_buffer[self.current_buffer].state_dict())

        return metrics

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.main_network.state_dict())
    
    def decay_epsilon(self):
        self.organism.epsilon = max(self.organism.epsilon_min, self.organism.epsilon * self.organism.epsilon_decay)

    def get_current_epsilon(self) -> float:
        return self.epsilon_buffer[self.current_buffer]

    def set_epsilon(self, new_epsilon: float):
        with self.network_lock:
            next_buffer = 1 - self.current_buffer
            self.epsilon_buffer[next_buffer] = new_epsilon