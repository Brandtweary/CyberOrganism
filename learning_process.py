import torch
import torch.multiprocessing as mp
import copy
from typing import Dict, Any, Tuple
import torch.nn.functional as F
import random


class LearningProcess:
    def __init__(self, input_queue, output_queue, main_network, target_network, learning_rate):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.main_network = main_network.to(self.device)
        self.target_network = target_network.to(self.device)
        self.learning_rate = learning_rate
        self.process = None
        self.training_steps = random.randint(0, 30)
        self.optimizer = None

    def start(self):
        self.optimizer = torch.optim.AdamW(self.main_network.parameters(), lr=self.learning_rate)
        self.process = mp.Process(target=self._run)
        self.process.start()

    def _run(self):
        while True:
            data = self.input_queue.get()
            if data is None:  # Termination signal
                break
            
            organism_state, experiences, total_reward = data
            metrics, weights, td_errors = self._learn(organism_state, experiences)
            self.output_queue.put((metrics, weights, td_errors, total_reward))

    def _learn(self, organism_state: Dict[str, Any], experiences: Any) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Dict[int, float]]:
        # Unpack parameters from organism_state
        gamma = organism_state['gamma']
        gradient_clip = organism_state['gradient_clip']
        target_update = organism_state['target_update']
        inference_update = organism_state['inference_update']

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
        if self.training_steps % target_update == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())

        # Check if inference network update is due
        weights = None
        if self.training_steps % inference_update == 0:
            weights = {k: v.cpu().detach().clone() for k, v in self.main_network.state_dict().items()}
        
        # Prepare metrics
        metrics = {
            "average_loss": loss.item(),
            "average_q_value": current_q_values.mean().item()
        }

        return metrics, weights, td_errors_dict

    def stop(self):
        if self.process:
            self.input_queue.put(None)  # Send termination signal
            self.process.join()  # Wait for the process to finish
            self.process = None

def setup_learning_process(input_queue, output_queue, main_network, target_network, learning_rate):
    # Create deep copies of the networks
    main_network_copy = copy.deepcopy(main_network)
    target_network_copy = copy.deepcopy(target_network)
    
    process = LearningProcess(input_queue, output_queue, main_network_copy, target_network_copy, learning_rate)
    process.start()
    return process