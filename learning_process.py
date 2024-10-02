import torch
import torch.multiprocessing as mp
import copy
from typing import Dict, Any, Tuple
import torch.nn.functional as F
import random
import uuid
import psutil
import os
from network_factory import create_DQN_network


class LearningProcess:
    def __init__(self, input_queue, output_queue, network_params, learning_rate, organism_id):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network_params = network_params
        self.learning_rate = learning_rate
        self.process = None
        self.training_steps = random.randint(0, 30)
        self.optimizer = None
        self.process_id = str(uuid.uuid4())  # Unique identifier for this process
        self.organism_id = organism_id
        self.main_network = None
        self.target_network = None

    def test_networks(self, input_size):
        # Test forward pass
        dummy_input = torch.randn(1, input_size).to(self.device)
        main_output = self.main_network(dummy_input)
        target_output = self.target_network(dummy_input)
        print(f"Organism {self.organism_id} initial forward pass:")
        print(f"Main network output: {main_output}")
        print(f"Target network output: {target_output}")
    
    def _log_network_initialization(self):
        print(f"Organism {self.organism_id} network initialization:")
        for name, network in [("Main", self.main_network), ("Target", self.target_network)]:
            device = next(network.parameters()).device  # Get the device of the network
            print(f"  {name} network is on device: {device}")
            for i, layer in enumerate(network.layers):
                if isinstance(layer, torch.nn.Linear):
                    weight_mean = layer.weight.data.mean().item()
                    weight_std = layer.weight.data.std().item()
                    print(f"  {name} network, layer {i}: weight mean = {weight_mean:.5f}, std = {weight_std:.5f}")

    def start(self):
        ctx = mp.get_context('spawn')
        self.process = ctx.Process(target=self._run)
        self.process.start()

    def _run(self):
        # Create networks inside the process
        self.main_network = create_DQN_network(**self.network_params).to(self.device)
        self.target_network = create_DQN_network(**self.network_params).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        
        self.optimizer = torch.optim.AdamW(self.main_network.parameters(), lr=self.learning_rate)
        
        # Log network state after process start
      #  self._log_network_initialization()
        
        # Test networks
      #  self.test_networks(self.network_params['input_size'])
        
        while True:
            data = self.input_queue.get()
            if data is None:  # Termination signal
                break
            
            organism_state, experiences, total_reward = data
            metrics, weights, td_errors = self._learn(organism_state, experiences)
            
            # Add memory usage to metrics
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            metrics['rss_memory'] = memory_info.rss / (1024 * 1024)  # RSS in MB
            metrics['vms_memory'] = memory_info.vms / (1024 * 1024)  # VMS in MB
            
            self.output_queue.put((metrics, weights, td_errors, total_reward))

    def _learn(self, organism_state: Dict[str, Any], experiences: Any) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Dict[int, float]]:
        # Unpack parameters from organism_state
        gamma = organism_state['gamma']
        gradient_clip = organism_state['gradient_clip']
        target_update = organism_state['target_update']
        inference_update = organism_state['inference_update']

        metrics = {}

        # Unpack experiences
        batch, idxs = experiences

        # Prepare batch data
        states, actions, rewards, next_states = zip(*batch)
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)

        # Compute Q-values for current states
        q_values = self.main_network(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            # Compute Q-values for next states
            next_q_values = self.target_network(next_states)

            # Get maximum Q-value for next states
            max_next_q_values = next_q_values.max(1)[0]

        # Compute expected Q-values
        expected_q_values = rewards + (gamma * max_next_q_values)

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=gradient_clip)
        self.optimizer.step()
        
        # Calculate TD errors
        with torch.no_grad():
            td_errors = abs(current_q_values - expected_q_values)
        td_errors_dict = {idx: error.item() for idx, error in zip(idxs, td_errors)}

        # Update target network if needed
        self.training_steps += 1
        target_update_counter = 0
        inference_update_counter = 0
        if self.training_steps % target_update == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())
            target_update_counter += 1

        # Check if inference network update is due
        weights = None
        if self.training_steps % inference_update == 0:
            weights = {k: v.cpu().detach().clone() for k, v in self.main_network.state_dict().items()}
            inference_update_counter += 1

        # Update metrics
        metrics.update({
            "average_loss": loss.item(),
            "average_q_value": current_q_values.mean().item(),
            "target_update_counter": target_update_counter,
            "inference_update_counter": inference_update_counter,
            "training_steps": self.training_steps
        })

        return metrics, weights, td_errors_dict

    def stop(self):
        if self.process:
            self.input_queue.put(None)  # Send termination signal
            self.process.join()  # Wait for the process to finish
            self.process = None

def setup_learning_process(input_queue, output_queue, network_params, learning_rate, organism_id):
    process = LearningProcess(input_queue, output_queue, network_params, learning_rate, organism_id)
    process.start()
    return process