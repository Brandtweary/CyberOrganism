import torch
import torch.multiprocessing as mp
from typing import Dict, Any, Tuple
import random
import uuid
import psutil
import os
from network_factory import setup_network_architecture

class LearningProcess:
    def __init__(self, input_queue, output_queue, network_params, organism_id):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.network_params = network_params
        self.process = None
        self.training_steps = random.randint(0, 30)
        self.process_id = str(uuid.uuid4())
        self.organism_id = organism_id
        self.architecture = {}
        self.device = network_params['device']

    def test_networks(self, input_size):
        dummy_input = torch.randn(1, input_size).to(self.device)
        main_output = self.main_network(dummy_input)
        target_output = self.target_network(dummy_input)
        print(f"Organism {self.organism_id} initial forward pass:")
        print(f"Main network output: {main_output}")
        print(f"Target network output: {target_output}")
    
    def log_network_initialization(self):
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
        architecture = setup_network_architecture(self.network_params)
        for key, value in architecture.items():
            setattr(self, key, value)
        self.architecture = architecture
        #self.test_networks(self.network_params['input_size'])
        #self.log_network_initialization()

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
        learn_callable = getattr(self, 'learn', None)
        if learn_callable:
            return learn_callable(self, organism_state, experiences)
        else:
            raise NotImplementedError("Learn function is not defined in the architecture")

    def stop(self):
        if self.process:
            self.input_queue.put(None)  # Send termination signal
            self.process.join()  # Wait for the process to finish
            self.process = None

def setup_learning_process(input_queue, output_queue, network_params, organism_id):
    process = LearningProcess(input_queue, output_queue, network_params, organism_id)
    process.start()
    return process