from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from typing import Any, Dict, Tuple
from enums import Action
from network_statistics import NetworkStatistics
from network_factory import create_neural_network
from state_snapshot import StateSnapshot
from custom_profiler import profiler
import queue
import threading
import copy
import numpy as np
import random
from collections import deque
import time


class ReinforcementLearningAlgorithm(ABC):
    """
    Abstract base class for deep reinforcement learning algorithms.

    This class serves as a template for implementing various deep RL algorithms
    such as DQN, DDPG, A2C, etc. It provides a common structure and interface
    for RL algorithms that use neural networks as function approximators.

    Subclasses should implement the abstract methods to define specific RL algorithms.
    Additionally, subclasses are responsible for initializing and managing any
    additional networks required for their particular implementation (e.g., target
    networks, critic networks) beyond the main network provided by this base class.
    """

    def __init__(self, organism: Any, action_mapping: Dict[int, str], network_params: Dict[str, Any]):
        self.organism = organism
        self.action_mapping = action_mapping
        self.network_params = network_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network_params['device'] = self.device
        self.network_stats = NetworkStatistics()

        self.process_pool = self.organism.sim_engine.process_pool
        self.process_pool.register_organism(self.organism.id, self.network_params)
        self.learn_input_queue, self.learn_output_queue = self.process_pool.get_organism_queues(self.organism.id)

        # New thread-safe queue for input processing
        self.input_queue = queue.Queue()

        # Inference network (CPU)
        self.inference_network = create_neural_network(self.network_params).to('cpu')
        self.inference_network.eval()
        self.inference_buffer = [self.inference_network, copy.deepcopy(self.inference_network)]
        self.current_inference_buffer = 0
        self.inference_buffer_lock = threading.Lock()

        # Shared counter for learning backlog
        self.learning_backlog = mp.Value('i', 0)

        # For storing metrics
        self.loss_history = deque(maxlen=100)
        self.q_value_history = deque(maxlen=100)
        self.target_update_counter = 0
        self.inference_update_counter = 0
        self.training_steps = 0

        # Threads for processing input and output queues
        self.running = True
        self.input_processing_thread = threading.Thread(target=self.process_input_queue, daemon=True)
        self.output_processing_thread = threading.Thread(target=self.process_output_queue, daemon=True)
        self.input_processing_thread.start()
        self.output_processing_thread.start()

    @abstractmethod
    def select_action(self, state: Any) -> Action:
        '''
        Ensure that the action is selected with the inference buffer lock
        '''
        pass

    @abstractmethod
    def _learn(organism_state: Dict[str, Any], experiences: Any, training_steps: int, architecture: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Dict[int, float]]:
        '''
        This method is implemented by the learning process and should not be called directly
        '''
        pass

    @abstractmethod
    def update_metrics(self, metrics, total_reward):
        '''
        Ensure that organism params are updated using add_param_diff for thread safety
        '''
        pass
    
    def queue_learn(self, organism_state: Dict[str, Any], experiences: Any):
        self.training_steps += 1
        self.input_queue.put((organism_state, experiences, self.training_steps))
        
        with self.learning_backlog.get_lock():
            self.learning_backlog.value += 1
    
    def get_learning_backlog(self):
        return self.learning_backlog.value

    def process_output_queue(self):
        while self.running:
            try:
                output = self.learn_output_queue.get(timeout=0.1)
                try:
                    self.process_learning_output(output)
                except Exception as e:
                    print(f"Exception in process_learning_output: {e}")
                    self.organism.sim_engine.stop_simulation = True
                    break
            except queue.Empty:
                time.sleep(0.01)  # Added sleep timer to prevent busy waiting
                continue
    
    def cleanup_threads(self):
        self.running = False
        self.input_processing_thread.join()
        self.output_processing_thread.join()
        self.process_pool.cleanup_organism(self.organism.id)

    def process_learning_output(self, output):
        metrics, weights, td_errors = output

        if weights is not None:
            self.update_inference_network(weights)

        self.update_metrics(metrics)
        self.update_replay_buffer_priorities(td_errors)

        with self.learning_backlog.get_lock():
            self.learning_backlog.value -= 1
    
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

    def process_input_queue(self):
        while self.running:
            try:
                input_data = self.input_queue.get(block=False)
                self.learn_input_queue.put(input_data)
            except queue.Empty:
                time.sleep(0.01)  # Short sleep to prevent busy waiting
            except Exception as e:
                print(f"Unexpected error in process_input_queue: {e}")
