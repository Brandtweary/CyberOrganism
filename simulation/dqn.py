import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from typing import Any, Dict, List, Tuple
from shared.enums import Action, RLAlgorithm
from shared.custom_profiler import profiler
from shared.summary_logger import summary_logger
from RL_algorithm import ReinforcementLearningAlgorithm
from state_snapshot import StateSnapshot


class DQN(ReinforcementLearningAlgorithm):
    def __init__(self, organism: Any, action_mapping: Dict[int, str], network_params: Dict[str, Any]):
        network_params['algorithm_type'] = RLAlgorithm.DQN.value
        network_params['learn'] = self._learn  # Pass the learn function
        network_params['select_action'] = self._select_action
        network_params['main_network'] = True
        network_params['target_network'] = True
        network_params['select_action_dependencies'] = ['torch', 'torch.nn.functional', 'random']
        network_params['learn_dependencies'] = ['torch', 'torch.nn.functional']

        # Then call the superclass initializer
        super().__init__(organism, action_mapping, network_params)  # call after network params are updated

    @staticmethod
    def _select_action(model, state, components):
        network_params = components['network_params']
        epsilon = network_params['epsilon']
        epsilon_min = network_params['epsilon_min']
        epsilon_decay = network_params['epsilon_decay']
        boltzmann_temperature = network_params['boltzmann_temperature']

        if random.random() < epsilon:
            action = random.choice(range(network_params['output_size']))
        else:
            with torch.inference_mode():
                q_values = model(state)
            # Boltzmann exploration
            probs = F.softmax(q_values / boltzmann_temperature, dim=-1)
            action = torch.multinomial(probs, 1).item()

        # Decay epsilon
        new_epsilon = max(epsilon_min, epsilon * epsilon_decay)
        components['network_params']['epsilon'] = new_epsilon

        return action

    def update_metrics(self, metrics):
        self.loss_history.append(metrics['average_loss'])
        self.q_value_history.append(metrics['average_q_value'])

        avg_loss = sum(self.loss_history) / len(self.loss_history)
        avg_q_value = sum(self.q_value_history) / len(self.q_value_history)

        self.target_update_counter += metrics['target_update_counter']
        self.inference_update_counter += metrics['inference_update_counter']

        self.organism.add_param_diff('average_loss', avg_loss)
        self.organism.add_param_diff('average_q_value', avg_q_value)
        self.organism.add_param_diff('target_update_counter', self.target_update_counter)
        self.organism.add_param_diff('inference_update_counter', self.inference_update_counter)
       
    @staticmethod
    def _learn(organism_state: Dict[str, Any], experiences: Any, training_steps: int, architecture: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Dict[int, float]]:
        gamma = organism_state['gamma']
        gradient_clip = organism_state['gradient_clip']
        target_update = organism_state['target_update']
        inference_update = organism_state['inference_update']

        main_network = architecture['main_network']
        target_network = architecture['target_network']
        optimizer = architecture['optimizer']
        device = architecture['device']

        metrics = {}

        # Unpack experiences
        batch, idxs = experiences

        # Prepare batch data
        states, actions, rewards, next_states = zip(*batch)

        states = torch.stack(states).to(device)
        next_states = torch.stack(next_states).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)

        
        # Compute Q-values for current states
        q_values = main_network(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Compute Q-values for next states
            next_q_values = target_network(next_states)
            # Get maximum Q-value for next states
            max_next_q_values = next_q_values.max(1)[0]

        # Compute expected Q-values
        expected_q_values = rewards + (gamma * max_next_q_values)

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(main_network.parameters(), max_norm=gradient_clip)
        optimizer.step()

        # Calculate TD errors
        with torch.no_grad():
            td_errors = abs(current_q_values - expected_q_values)
        td_errors_dict = {idx: error.item() for idx, error in zip(idxs, td_errors)}

        # Update target network if needed
        training_steps += 1
        target_update_counter = 0
        inference_update_counter = 0
        if training_steps % target_update == 0:
            target_network.load_state_dict(main_network.state_dict())
            target_update_counter += 1

        # Check if inference network update is due
        weights = None
        if training_steps % inference_update == 0:
            weights = {k: v.cpu().detach().clone() for k, v in main_network.state_dict().items()}
            inference_update_counter += 1

        # Update metrics
        metrics.update({
            "average_loss": loss.item(),
            "average_q_value": current_q_values.mean().item(),
            "target_update_counter": target_update_counter,
            "inference_update_counter": inference_update_counter,
            "training_steps": training_steps
        })

        return metrics, weights, td_errors_dict
