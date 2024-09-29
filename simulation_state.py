import psutil
import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates

class SimulationState:
    def __init__(self, simulation_engine, ui):
        self.sim_engine = simulation_engine
        self.ui = ui
        self.current_state = simulation_engine.current_state
        self.test_organism = simulation_engine.test_organism
        self.input_parameters = {param: getattr(self.test_organism, param) for param in self.test_organism.input_parameters}
        self.display_parameters = {param: getattr(self.test_organism, param) for param in self.test_organism.display_parameters}
        self.cpu_usage = psutil.cpu_percent()
        self.memory_usage = psutil.virtual_memory().percent
        self.available_memory = psutil.virtual_memory().available / (1024 * 1024)
        self.gpu_usage = 0.0
        self.framerate = 0.0
        self.learn_queue_size = 0
        self.deceased_organisms = self.sim_engine.deceased_organisms

        # Summary statistics
        self.time_low_loss = 0
        self.start_time = time.time()
        self.total_time = 0

        # Add network statistics
        self.network_stats = self.test_organism.RL_algorithm.network_stats.stats
        self.network_record_stats = self.test_organism.RL_algorithm.network_stats.record_stats

        # Add new attributes for storing average times
        self.avg_total_frame_times = []

        self.ui_update_timer = 0
        self.last_updated_section = -1
        self.ui_update_interval = 0.1

        self.frame_count = 0
        self.loading_frames = 60

        nvmlInit()
        self.gpu_handle = nvmlDeviceGetHandleByIndex(0)

    def generate_organism_stats(self):
        stats = {}
        for param, value in self.display_parameters.items():
            formatted_name = self.format_parameter_name(param)
            formatted_value = f"{value:.3f}" if isinstance(value, float) else value
            stats[formatted_name] = formatted_value
        return stats

    def generate_performance_stats(self):
        return {
            "CPU Usage": f"{self.cpu_usage:.1f}%",
            "Memory Usage": f"{self.memory_usage:.1f}%",
            "GPU Usage": f"{self.gpu_usage:.1f}%",
            "Learn Queue": str(self.learn_queue_size),
            "Organism Count": str(self.num_organisms),
            "Deceased Organisms": str(self.deceased_organisms),
            "Item Count": str(self.num_items),
            "FPS": f"{self.framerate:.1f}",
            "Simulation Time": f"{int(self.total_time)} seconds",
        }

    def generate_network_stats(self):
        stats = {}
        for stat_type, stat_info in self.network_record_stats.items():
            if stat_info['record']:
                for stat_name in stat_info['stat_names']:
                    if stat_name in self.network_stats:
                        values = self.network_stats[stat_name]
                        if values:
                            value = values[-1]
                            formatted_value = f"{value:.3f}" if isinstance(value, float) else value
                            stats[f"{stat_type} - {stat_name}"] = formatted_value
        return stats
    
    def generate_training_metrics(self):
        return {
            "Average Reward": f"{self.test_organism.average_reward:.3f}",
            "Average Loss": f"{self.test_organism.average_loss:.3f}",
            "Average Q-Value": f"{self.test_organism.average_q_value:.3f}"
        }

    def update(self):
        self.frame_count += 1
        
        self.sim_engine.update_simulation()
        self.current_state = self.sim_engine.current_state
        self.input_parameters = {param: getattr(self.test_organism, param) for param in self.test_organism.input_parameters}
        self.display_parameters = {param: getattr(self.test_organism, param) for param in self.test_organism.display_parameters}
        
        # Update summary statistics
        current_loss = self.test_organism.average_loss
        if current_loss < 0.5:
            self.time_low_loss += self.sim_engine.UPDATE_INTERVAL
        self.total_time = time.time() - self.start_time
        
        # Update performance metrics
        self.cpu_usage = psutil.cpu_percent()
        self.memory_usage = psutil.virtual_memory().percent
        utilization = nvmlDeviceGetUtilizationRates(self.gpu_handle)
        self.gpu_usage = utilization.gpu
        self.learn_queue_size = sum(organism.RL_algorithm.learn_queue.qsize() for organism in self.sim_engine.organisms)
        self.deceased_organisms = self.sim_engine.deceased_organisms

        self.num_organisms = len(self.sim_engine.organisms)
        self.num_items = len(self.sim_engine.items)

        # Update network statistics
        self.network_stats = self.test_organism.RL_algorithm.network_stats.stats
        self.network_record_stats = self.test_organism.RL_algorithm.network_stats.record_stats

        # Update UI
        if self.frame_count > self.loading_frames - 2:
            self.update_ui()

    def update_ui(self):
        if self.last_updated_section == -1:
            # First update: update all sections
            organism_stats = self.generate_organism_stats()
            performance_stats = self.generate_performance_stats()
            training_metrics = self.generate_training_metrics()
            action_distribution = self.test_organism.action_distribution
            self.ui.update_left_sidebar(organism_stats, performance_stats, training_metrics, action_distribution)
            self.last_updated_section = 2  # Set to 2 so next update starts with 0
        else:
            self.ui_update_timer += self.sim_engine.UPDATE_INTERVAL
            if self.ui_update_timer >= self.ui_update_interval:
                self.ui_update_timer = 0
                self.last_updated_section = (self.last_updated_section + 1) % 3

                if self.last_updated_section == 0:
                    organism_stats = self.generate_organism_stats()
                    self.ui.update_left_sidebar(organism_stats, None, None, None)
                elif self.last_updated_section == 1:
                    performance_stats = self.generate_performance_stats()
                    self.ui.update_left_sidebar(None, performance_stats, None, None)
                else:
                    training_metrics = self.generate_training_metrics()
                    action_distribution = self.test_organism.action_distribution
                    self.ui.update_left_sidebar(None, None, training_metrics, action_distribution)

    @staticmethod
    def format_parameter_name(name):
        if name is None:
            return "None"
        return ' '.join(word.capitalize() for word in str(name).split('_'))