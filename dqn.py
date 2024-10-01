import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import random
import copy
from typing import Any, Dict, List, Tuple
from enums import Action
from RL_algorithm import ReinforcementLearningAlgorithm
from state_snapshot import StateSnapshot
from custom_profiler import profiler
import learning_process
import numpy as np
import threading
from collections import deque
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
        
        # Main and target networks (GPU)
        self.main_network = self.create_network().to(self.device)
        self.target_network = self.create_network().to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()
        
        # Inference network (CPU)
        self.inference_network = self.create_network().to('cpu')
        self.inference_network.load_state_dict(self.main_network.state_dict())
        self.inference_network.eval()
        
        # Inference network buffer
        self.inference_buffer = [self.inference_network, copy.deepcopy(self.inference_network)]
        self.current_inference_buffer = 0
        
        # Optimizer
        self.optimizer = self.create_optimizer()
        
        self.setup_learning_process(self.learning_rate)

        # Shared counter for learning backlog
        self.learning_backlog = mp.Value('i', 0)

        self.inference_buffer_lock = threading.Lock()
        self.output_processing_thread = threading.Thread(target=self._process_output_queue, daemon=True)
        self.output_processing_thread.start()

        # For storing metrics
        self.reward_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        self.q_value_history = deque(maxlen=100)
        

    def create_network(self) -> nn.Module:
        return DQNNetwork(self.input_size, self.hidden_size, self.output_size, self.hidden_layers)

    def create_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(self.main_network.parameters(), lr=self.learning_rate)

    def setup_learning_process(self, learning_rate):
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        
        self.learning_process = learning_process.setup_learning_process(
            self.input_queue, 
            self.output_queue, 
            self.main_network,
            self.target_network,
            learning_rate
        )
    
    def cleanup(self):
        if self.learning_process:
            self.learning_process.stop()
            self.learning_process = None

    def select_action(self, state: torch.Tensor) -> int:
        if random.random() < self.organism.epsilon:
            self.decay_epsilon()
            return random.randint(0, self.output_size - 1)
        else:
            with torch.no_grad():
                with self.inference_buffer_lock:
                    current_buffer = self.current_inference_buffer
                q_values = self.inference_buffer[current_buffer](state)
            self.decay_epsilon()
            return self.boltzmann_exploration(q_values)

    @profiler.profile("queue_learn")
    def queue_learn(self, state_snapshot: StateSnapshot, total_reward: float, experiences: Any):
        organism_state = state_snapshot.get_state(self.organism.id)
        self.input_queue.put((organism_state.copy(), experiences, total_reward))
        
        with self.learning_backlog.get_lock():
            self.learning_backlog.value += 1
    
    def get_learning_backlog(self):
        return self.learning_backlog.value
    
    def _process_output_queue(self):
        while True:
            try:
                output = self.output_queue.get(timeout=0.1)
                self._process_learning_output(output)
            except queue.Empty:
                continue

    def _process_learning_output(self, output):
        metrics, weights, td_errors, total_reward = output

        if weights is not None:
            self.update_inference_network(weights)

        self._update_metrics(metrics, total_reward)
        self.update_replay_buffer_priorities(td_errors)

        with self.learning_backlog.get_lock():
            self.learning_backlog.value -= 1

    def _update_metrics(self, metrics, total_reward):
        self.reward_history.append(total_reward)
        self.loss_history.append(metrics['average_loss'])
        self.q_value_history.append(metrics['average_q_value'])

        avg_reward = sum(self.reward_history) / len(self.reward_history)
        avg_loss = sum(self.loss_history) / len(self.loss_history)
        avg_q_value = sum(self.q_value_history) / len(self.q_value_history)

        self.organism.add_param_diff('average_reward', avg_reward)
        self.organism.add_param_diff('average_loss', avg_loss)
        self.organism.add_param_diff('average_q_value', avg_q_value)

    def update_inference_network(self, weights: Dict[str, torch.Tensor]):
        new_inference_network = self.inference_buffer[1 - self.current_inference_buffer]
        new_inference_network.load_state_dict(weights)
        new_inference_network.to('cpu')
        new_inference_network.eval()
        with self.inference_buffer_lock:
            self.current_inference_buffer = 1 - self.current_inference_buffer

    def update_replay_buffer_priorities(self, td_errors_dict: Dict[int, float]):
        idxs = list(td_errors_dict.keys())
        priorities = np.array(list(td_errors_dict.values()))
        self.organism.replay_buffer.update_priorities(idxs, priorities)

    def decay_epsilon(self):
        self.organism.epsilon = max(self.organism.epsilon_min, self.organism.epsilon * self.organism.epsilon_decay)

    def boltzmann_exploration(self, q_values: torch.Tensor) -> int:
        probabilities = F.softmax(q_values / self.organism.boltzmann_temperature, dim=0)
        action_index = torch.multinomial(probabilities, 1).item()
        return action_index
    
    def _learn(self, organism_state: Dict[str, Any], experiences: Any):
        pass  # dummy implementation

    # def _learn(self, organism_state: Dict[str, Any], experiences: Any) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Dict[int, float]]:
    #     # Unpack parameters from organism_state
    #     gamma = organism_state['gamma']
    #     gradient_clip = organism_state['gradient_clip']
    #     target_update = organism_state['target_update']
    #     inference_update = organism_state['inference_update']

    #     # Unpack experiences
    #     batch, idxs = experiences

    #     # Prepare batch data
    #     states, actions, rewards, next_states = zip(*batch)
    #     states = torch.stack(states).to(self.device)
    #     next_states = torch.stack(next_states).to(self.device)
    #     rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
    #     actions = torch.tensor(actions, dtype=torch.long).to(self.device)

    #     # Compute Q-values for current states
    #     q_values = self.main_network(states)
    #     current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    #     with torch.no_grad():
    #         # Compute Q-values for next states
    #         next_q_values = self.target_network(next_states)
    #         # Get maximum Q-value for next states
    #         max_next_q_values = next_q_values.max(1)[0]

    #     # Compute expected Q-values
    #     expected_q_values = rewards + (gamma * max_next_q_values)

    #     # Compute loss
    #     loss = F.smooth_l1_loss(current_q_values, expected_q_values)

    #     # Optimize the model
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=gradient_clip)
    #     self.optimizer.step()

    #     # Calculate TD errors
    #     with torch.no_grad():
    #         td_errors = abs(current_q_values - expected_q_values)
    #     td_errors_dict = {idx: error.item() for idx, error in zip(idxs, td_errors)}

    #     # Update target network if needed
    #     self.training_steps += 1
    #     if self.training_steps % target_update == 0:
    #         self.target_network.load_state_dict(self.main_network.state_dict())

    #     # Check if inference network update is due
    #     weights = None
    #     if self.training_steps % inference_update == 0:
    #         weights = self.main_network.state_dict()
        
    #     # Prepare metrics
    #     metrics = {
    #         "average_loss": loss.item(),
    #         "average_q_value": current_q_values.mean().item()
    #     }

    #     return metrics, weights, td_errors_dict
