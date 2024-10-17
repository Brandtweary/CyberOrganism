import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import time
import json
from shared.custom_profiler import profiler
from shared.network_factory import create_neural_network
from typing import List, Dict, Any

def setup_network_params() -> Dict[str, Any]:
    return {
        'input_size': 7,  # Adjust this based on your actual input size
        'hidden_size': 64,
        'output_size': 5,  # Adjust this based on your actual number of actions
        'hidden_layers': 2,
        'learning_rate': 0.001,
        'device': 'cpu',
        'algorithm_type': 'dqn',
        'main_network': False,
        'target_network': False
    }

def load_dummy_states(file_path: str) -> List[torch.Tensor]:
    with open(file_path, 'r') as f:
        states = json.load(f)
    return [torch.tensor(state, dtype=torch.float32) for state in states]

def boltzmann_exploration(q_values: torch.Tensor, temperature: float) -> int:
    probs = torch.softmax(q_values / temperature, dim=0)
    return torch.multinomial(probs, 1).item()

@profiler.profile("select_action")
def select_action(state: torch.Tensor, network: torch.nn.Module, temperature: float) -> int:
    with torch.inference_mode():
        with profiler.profile_section("select_action", "model_inference"):
            q_values = network(state)
    
    with profiler.profile_section("select_action", "boltzmann_exploration"):
        return boltzmann_exploration(q_values, temperature)

def run_inference_test(states: List[torch.Tensor], networks: List[torch.nn.Module], temperature: float):
    start_time = time.time()
    actions_per_second = 30
    interval = 1.0 / actions_per_second
    last_print_time = time.time()
    total_states = len(states)
    skipped_calls = 10

    for i, state in enumerate(states):
        for network in networks:
            action = select_action(state, network, temperature)

        if i == skipped_calls - 1:
            profiler.reset_stats()
            print(f"Profiler stats reset after {skipped_calls} calls")
        
        # Simulate real-time processing
        elapsed = time.time() - start_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
        start_time = time.time()

        # Print remaining states every 1 second
        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            remaining_states = total_states - i - 1
            print(f"States remaining: {remaining_states}")
            last_print_time = current_time

def load_dummy_weights(networks: List[torch.nn.Module], weights_file: str):
    state_dict = torch.load(weights_file)
    for network in networks:
        network.load_state_dict(state_dict)
    print(f"Loaded weights from {weights_file} to {len(networks)} networks")

def main():
    """
    Main function to run the inference test. If state tensors are the wrong size for the network,
    delete the dummy states, re-run the simulation for a minimum of 120 seconds (30 * frame skip value),
    and try again. If the weights can't be loaded into the network, adjust the network params to reflect
    the current network architecture of the organism.
    """
    network_params = setup_network_params()
    num_organisms = 20
    networks = [create_neural_network(network_params) for _ in range(num_organisms)]
    
    # Load dummy weights onto every network
    load_dummy_weights(networks, 'testing/test_data/dummy_network_weights.pth')
    
    for network in networks:
        network.eval()

    # Replace the dummy state creation with loading from file
    states = load_dummy_states('testing/test_data/dummy_states.json')
    temperature = 0.2  # Adjust this based on your desired Boltzmann temperature

    run_inference_test(states, networks, temperature)

    print(profiler.get_performance_stats())

if __name__ == "__main__":
    main()
