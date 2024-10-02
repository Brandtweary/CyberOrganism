import torch
import torch.nn as nn
from typing import List

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

def create_DQN_network(input_size: int, hidden_size: int, output_size: int, hidden_layers: int) -> nn.Module:
    return DQNNetwork(input_size, hidden_size, output_size, hidden_layers)
