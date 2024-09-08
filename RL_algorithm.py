from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, Tuple
from enums import Action
from training_statistics import TrainingStatistics


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

    def __init__(self, organism: Any, action_mapping: Dict[int, str], input_size: int, hidden_size: int, output_size: int, hidden_layers: int, learning_rate: float):
        self.organism = organism
        self.action_mapping = action_mapping
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        
        self.main_network = self.create_network()
        self.optimizer = self.create_optimizer()
        self.training_stats = TrainingStatistics(self.main_network, self.optimizer)

    @abstractmethod
    def create_network(self) -> nn.Module:
        pass

    @abstractmethod
    def create_optimizer(self) -> optim.Optimizer:
        pass

    @abstractmethod
    def select_action(self, state: torch.Tensor) -> Action:
        pass

    @abstractmethod
    def learn(self) -> Dict[str, float]:
        pass

    def get_network_parameters(self) -> Dict[str, torch.Tensor]:
        return {name: param.data for name, param in self.main_network.named_parameters()}

    def load_network_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        for name, param in self.main_network.named_parameters():
            if name in parameters:
                param.data.copy_(parameters[name])
