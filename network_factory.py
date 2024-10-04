import torch
import torch.nn as nn
from typing import Dict, Any


class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, hidden_layers: int):
        super(NeuralNetwork, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

def create_neural_network(network_params: Dict[str, Any]) -> nn.Module:
    input_size = network_params['input_size']
    hidden_size = network_params['hidden_size']
    output_size = network_params['output_size']
    hidden_layers = network_params['hidden_layers']
    device = network_params['device']

    network = NeuralNetwork(input_size, hidden_size, output_size, hidden_layers).to(device)
    return network

def setup_network_architecture(network_params: Dict[str, Any]) -> Dict[str, Any]:
    architecture: Dict[str, Any] = {}
    learning_rate = network_params['learning_rate']
    
    # Create networks based on flags in network_params
    if network_params.get('main_network'):
        architecture['main_network'] = create_neural_network(network_params)
    if network_params.get('target_network'):
        architecture['target_network'] = create_neural_network(network_params)
    if 'learn' in network_params:
        architecture['learn'] = network_params['learn']

    # Create optimizer if main_network exists
    if 'main_network' in architecture:
        architecture['optimizer'] = torch.optim.AdamW(architecture['main_network'].parameters(), lr=learning_rate)
    
    architecture['device'] = network_params['device']

    return architecture