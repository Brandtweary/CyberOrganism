from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, Tuple
from enums import Action
from network_statistics import NetworkStatistics
from learning_process import LearningProcess
import queue
import threading
import copy
import random


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

    @abstractmethod
    def select_action(self, state: Any) -> Action:
        '''
        Ensure that inferences are made with the network lock
        '''
        pass

    @abstractmethod
    def _learn(process: LearningProcess, organism_state: Dict[str, Any], experiences: Any) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Dict[int, float]]:
        '''
        Ensure that network params are updated with the network lock
        '''
        pass

    def get_network_parameters(self) -> Dict[str, Any]:
        return {name: param.data for name, param in self.main_network.named_parameters()}

    def load_network_parameters(self, parameters: Dict[str, Any]) -> None:
        for name, param in self.main_network.named_parameters():
            if name in parameters:
                param.data.copy_(parameters[name])
