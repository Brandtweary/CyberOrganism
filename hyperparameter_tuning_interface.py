from simulation_engine import SimulationEngine
from organism import Organism
import torch

class HyperparameterTuningInterface:
    def __init__(self):
        self.sim_engine = SimulationEngine()
        initial_x, initial_y = self.sim_engine.GRID_SIZE // 2, self.sim_engine.GRID_SIZE // 2
        self.organism = self.sim_engine.create_organism(Organism, initial_x, initial_y)
        self.update_hyperparameters()

    def update_hyperparameters(self):
        # RL hyperparameters
        self.discount_factor = getattr(self.organism, 'gamma', None)
        self.epsilon = getattr(self.organism, 'epsilon', None)
        self.epsilon_decay = getattr(self.organism, 'epsilon_decay', None)
        self.epsilon_min = getattr(self.organism, 'epsilon_min', None)
        self.replay_buffer_size = getattr(getattr(self.organism, 'experience_buffer', None), 'maxlen', None)
        self.batch_size = getattr(self.organism, 'batch_size', None)
        self.target_update_frequency = getattr(self.organism, 'target_update', None)
        self.training_frequency = 1  # This is not explicitly defined, assumed to be every step
        self.initial_random_steps = 0  # This is not explicitly defined in the current implementation

        # Network architecture
        self.input_size = getattr(self.organism, 'input_size', None)
        self.hidden_size = getattr(self.organism, 'hidden_size', None)
        self.output_size = getattr(self.organism, 'output_size', None)
        self.hidden_layers = getattr(self.organism, 'hidden_layers', None)

        # Optimizer parameters
        self.optimizer_params = self.organism.RL_neural_network.optimizer.defaults
        self.learning_rate = self.organism.learning_rate
        
        # Environment parameters
        self.grid_size = getattr(self.sim_engine, 'GRID_SIZE', None)
        self.num_items = len(getattr(self.sim_engine, 'items', []))
        self.initial_energy = getattr(self.organism, 'energy', None)
        self.initial_nutrition = getattr(self.organism, 'nutrition', None)
        self.energy_consumption = getattr(self.organism, 'energy_consumption', None)
        self.nutrition_consumption = getattr(self.organism, 'nutrition_consumption', None)
        self.food_replenishment_rate = getattr(self.sim_engine, 'FOOD_REPLENISHMENT_RATE', None)

        # Organism-specific parameters
        self.sensory_range = getattr(self.organism, 'detection_radius', None)
        self.movement_speed = getattr(self.organism, 'movement_speed', None)
        self.metabolism_rate = None  # This is not explicitly defined in the current implementation

        # Prioritized Experience Replay parameters (not implemented)
        self.per_alpha = None
        self.per_beta = None
        self.per_epsilon = None

        # Multi-step learning parameter (not implemented)
        self.n_step_returns = None

        # New parameters (not yet implemented, set to None)
        # Reward scaling factor: Used to normalize rewards, important for stable learning
        self.reward_scaling_factor = getattr(self.organism, 'reward_scaling_factor', None)

        # Clipping value for gradients: Prevents exploding gradients during training
        self.gradient_clip_value = getattr(self.organism, 'gradient_clip_value', None)

        # Activation functions: Defines non-linearity in neural networks
        self.activation_function = getattr(self.organism.RL_neural_network.main_network, 'activation_function', None)

        # Initialization method: Affects initial weights of the neural network
        self.weight_init_method = getattr(self.organism.RL_neural_network.main_network, 'weight_init_method', None)

        # Learning rate decay schedule: Adjusts learning rate over time
        self.lr_decay_schedule = getattr(self.organism.optimizer, 'lr_scheduler', None)

        # Entropy coefficient: Encourages exploration in policy gradient methods
        self.entropy_coefficient = getattr(self.organism, 'entropy_coefficient', None)

        # Exploration noise parameters: Used in some continuous action space algorithms
        self.exploration_noise_params = getattr(self.organism, 'exploration_noise_params', None)

        # Maximum episode length: Limits the length of each training episode
        self.max_episode_length = getattr(self.sim_engine, 'MAX_EPISODE_LENGTH', None)

        # Number of parallel environments: For parallelized training
        self.num_parallel_envs = getattr(self.sim_engine, 'NUM_PARALLEL_ENVS', None)

        # Normalization parameters: For normalizing state/action spaces
        self.state_normalization_params = getattr(self.organism, 'state_normalization_params', None)
        self.action_normalization_params = getattr(self.organism, 'action_normalization_params', None)

    def print_all_hyperparameters(self):
        all_params = vars(self)
        hyperparameters = {param: value for param, value in all_params.items() if param not in ['SimulationEngine', 'organism']}
        
        print("=== Hyperparameters ===")
        print(f"Total number of hyperparameters: {len(hyperparameters)}")
        print()
        
        for param, value in hyperparameters.items():
            print(f"{param}: {value}")

if __name__ == "__main__":
    torch.manual_seed(0)  # For reproducibility
    hti = HyperparameterTuningInterface()
    hti.print_all_hyperparameters()
