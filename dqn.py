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
from learning_process import setup_learning_process, LearningProcess
import numpy as np
import threading
from collections import deque
import queue
from summary_logger import summary_logger
from network_factory import create_neural_network
from enums import RLAlgorithm


class DQN(ReinforcementLearningAlgorithm):
    def __init__(self, organism: Any, action_mapping: Dict[int, str], network_params: Dict[str, Any]):
        super().__init__(organism, action_mapping, network_params)
        self.network_params['algorithm_type'] = RLAlgorithm.DQN.value
        self.network_params['learn'] = self._learn  # Pass the learn function
        self.network_params['main_network'] = True
        self.network_params['target_network'] = True
        
        # Inference network (CPU)
        self.inference_network = create_neural_network(self.network_params).to('cpu')
        self.inference_network.eval()
        
        # Inference network buffer
        self.inference_buffer = [self.inference_network, copy.deepcopy(self.inference_network)]
        self.current_inference_buffer = 0
        
        self._setup_learning_process()

        # Shared counter for learning backlog
        self.learning_backlog = mp.Value('i', 0)

        self.inference_buffer_lock = threading.Lock()
        self.output_processing_thread = threading.Thread(target=self._process_output_queue, daemon=True)
        self.output_processing_thread.start()

        # For storing metrics
        self.reward_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        self.q_value_history = deque(maxlen=100)
        self.target_update_counter = 0
        self.inference_update_counter = 0
        self.learn_counter = 0

    def _setup_learning_process(self):
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        
        self.learning_process = setup_learning_process(
            self.input_queue, 
            self.output_queue, 
            self.network_params,
            self.organism.id
        )
    
    def cleanup(self):
        if self.learning_process:
            self.learning_process.stop()
            self.learning_process = None

    def select_action(self, state: torch.Tensor) -> int:
        if random.random() < self.organism.epsilon:
            self.decay_epsilon()
            return random.choice(list(self.action_mapping.keys()))
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
                try:
                    self._process_learning_output(output)
                except Exception as e:
                    print(f"Exception in _process_learning_output: {e}")
                    self.organism.sim_engine.stop_simulation = True
                    break
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

        self.target_update_counter += metrics['target_update_counter']
        self.inference_update_counter += metrics['inference_update_counter']
        self.learn_counter += 1

        self.organism.add_param_diff('average_reward', avg_reward)
        self.organism.add_param_diff('average_loss', avg_loss)
        self.organism.add_param_diff('average_q_value', avg_q_value)
        self.organism.add_param_diff('target_update_counter', self.target_update_counter)
        self.organism.add_param_diff('inference_update_counter', self.inference_update_counter)
        self.organism.add_param_diff('learn_counter', self.learn_counter)
        self.organism.add_param_diff('learning_rss_memory', metrics['rss_memory'])
        self.organism.add_param_diff('learning_vms_memory', metrics['vms_memory'])

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
    
    @staticmethod
    def _learn(process: LearningProcess, organism_state: Dict[str, Any], experiences: Any) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Dict[int, float]]:
        gamma = organism_state['gamma']
        gradient_clip = organism_state['gradient_clip']
        target_update = organism_state['target_update']
        inference_update = organism_state['inference_update']

        metrics = {}

        # Unpack experiences
        batch, idxs = experiences

        # Prepare batch data
        states, actions, rewards, next_states = zip(*batch)
        states = torch.stack(states).to(process.device)
        next_states = torch.stack(next_states).to(process.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(process.device)
        actions = torch.tensor(actions, dtype=torch.long).to(process.device)

        # Compute Q-values for current states
        q_values = process.main_network(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Compute Q-values for next states
            next_q_values = process.target_network(next_states)
            # Get maximum Q-value for next states
            max_next_q_values = next_q_values.max(1)[0]

        # Compute expected Q-values
        expected_q_values = rewards + (gamma * max_next_q_values)

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        # Optimize the model
        process.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(process.main_network.parameters(), max_norm=gradient_clip)
        process.optimizer.step()

        # Calculate TD errors
        with torch.no_grad():
            td_errors = abs(current_q_values - expected_q_values)
        td_errors_dict = {idx: error.item() for idx, error in zip(idxs, td_errors)}

        # Update target network if needed
        process.training_steps += 1
        target_update_counter = 0
        inference_update_counter = 0
        if process.training_steps % target_update == 0:
            process.target_network.load_state_dict(process.main_network.state_dict())
            target_update_counter += 1

        # Check if inference network update is due
        weights = None
        if process.training_steps % inference_update == 0:
            weights = {k: v.cpu().detach().clone() for k, v in process.main_network.state_dict().items()}
            inference_update_counter += 1

        # Update metrics
        metrics.update({
            "average_loss": loss.item(),
            "average_q_value": current_q_values.mean().item(),
            "target_update_counter": target_update_counter,
            "inference_update_counter": inference_update_counter,
            "training_steps": process.training_steps
        })

        return metrics, weights, td_errors_dict