import psutil
import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
from concurrent.futures import ThreadPoolExecutor
from custom_profiler import profiler
from threading import Thread
from queue import Queue
from PySide6.QtCore import QThread, Signal, QObject



class SimulationState:
    def __init__(self, simulation_engine, ui):
        # Core simulation attributes
        self.sim_engine = simulation_engine
        self.ui = ui
        self.current_state = simulation_engine.current_state
        self.test_organism = simulation_engine.test_organism

        # Organism parameters
        self.input_parameters = {param: getattr(self.test_organism, param) for param in self.test_organism.input_parameters}
        self.display_parameters = {param: getattr(self.test_organism, param) for param in self.test_organism.display_parameters}

        # Performance metrics
        self.cpu_usage = psutil.cpu_percent()
        self.memory_usage = psutil.virtual_memory().percent
        self.available_memory = psutil.virtual_memory().available / (1024 * 1024)
        self.gpu_usage = 0.0
        self.framerate = 0.0
        nvmlInit()
        self.gpu_handle = nvmlDeviceGetHandleByIndex(0)
        self.learning_backlog = 0

        # Simulation statistics
        self.deceased_organisms = self.sim_engine.deceased_organisms
        self.num_organisms = len(self.sim_engine.organisms)
        self.num_items = len(self.sim_engine.items)
        self.all_organism_stats = {}  # Renamed from simulation_stats

        # Training metrics
        self.time_low_loss = 0
        self.start_time = time.time()
        self.total_time = 0
        self.network_stats = self.test_organism.RL_algorithm.network_stats.stats
        self.network_record_stats = self.test_organism.RL_algorithm.network_stats.record_stats
        self.action_distribution = self.test_organism.action_distribution

        # Framerate Calculation
        self.avg_total_frame_times = []
        self.max_frame_time = 0

        self.performance_updater = PerformanceUpdater(self)
        self.ui_updater = UIUpdater(self)
        self.ui_updater.update_signal.connect(self.ui.update_left_sidebar)
        self.ui_updater.start()        
        self.performance_updater.start()

    @profiler.profile("update_simulation_state")
    def update(self):        
        with profiler.profile_section("update_simulation_state", "update_simulation"):
            # Update core simulation
            self.sim_engine.update_simulation()
            self.current_state = self.sim_engine.current_state
            self.test_organism = self.sim_engine.test_organism

        with profiler.profile_section("update_simulation_state", "organism_parameters"):            # Update organism parameters
            self.input_parameters = {param: getattr(self.test_organism, param) for param in self.test_organism.input_parameters}
            self.display_parameters = {param: getattr(self.test_organism, param) for param in self.test_organism.display_parameters}
        
        with profiler.profile_section("update_simulation_state", "performance_metrics"):
            # Update performance metrics
            self.update_performance_metrics()

        with profiler.profile_section("update_simulation_state", "simulation_statistics"):
            # Update simulation statistics
            self.update_simulation_statistics()

        with profiler.profile_section("update_simulation_state", "training_metrics"):
            # Update training metrics
            self.update_training_metrics()

    def update_performance_metrics(self):
        if not self.performance_updater.queue.empty():
            self.cpu_usage, self.memory_usage, self.gpu_usage, self.learning_backlog = self.performance_updater.queue.get()

    def update_simulation_statistics(self):
        self.deceased_organisms = self.sim_engine.deceased_organisms
        self.num_organisms = len(self.sim_engine.organisms)
        self.num_items = len(self.sim_engine.items)
        
        # Update organism counters
        self.all_organism_stats = []
        for organism in self.sim_engine.organisms:
            org_id = str(organism.id)[:4]
            org_marker = '*' if organism is self.test_organism else ''
            stat_name = f"Organism {org_id}{org_marker}"
            stat_value = f"{organism.RL_algorithm.target_update_counter}/{organism.RL_algorithm.inference_update_counter}/{organism.learn_counter}"
            self.all_organism_stats.append((stat_name, stat_value))

    def update_training_metrics(self):
        current_loss = self.test_organism.average_loss
        if current_loss < 0.1:
            self.time_low_loss += self.sim_engine.UPDATE_INTERVAL
        self.total_time = time.time() - self.start_time
        self.network_stats = self.test_organism.RL_algorithm.network_stats.stats
        self.network_record_stats = self.test_organism.RL_algorithm.network_stats.record_stats
        self.action_distribution = self.test_organism.action_distribution

    def generate_organism_stats(self):
        return [
            (self.format_parameter_name(param), f"{value:.3f}" if isinstance(value, float) else value)
            for param, value in self.display_parameters.items()
        ]

    def generate_performance_stats(self):
        return [
            ("CPU Usage", f"{self.cpu_usage:.1f}%"),
            ("Memory Usage", f"{self.memory_usage:.1f}%"),
            ("GPU Usage", f"{self.gpu_usage:.1f}%"),
            ("Organism Count", str(self.num_organisms)),
            ("Deceased Organisms", str(self.deceased_organisms)),
            ("Item Count", str(self.num_items)),
            ("Learning Backlog", str(self.learning_backlog)),
            ("FPS", f"{self.framerate:.1f}"),
            ("Simulation Time", f"{int(self.total_time)} seconds")
        ]

    def generate_network_stats(self):
        stats = []
        for stat_type, stat_info in self.network_record_stats.items():
            if stat_info['record']:
                for stat_name in stat_info['stat_names']:
                    if stat_name in self.network_stats:
                        values = self.network_stats[stat_name]
                        if values:
                            value = values[-1]
                            formatted_value = f"{value:.3f}" if isinstance(value, float) else value
                            stats.append((f"{stat_type} - {stat_name}", formatted_value))
        return stats
    
    def generate_training_stats(self):
        return [
            ("Average Reward", f"{self.test_organism.average_reward:.3f}"),
            ("Average Loss", f"{self.test_organism.average_loss:.3f}"),
            ("Average Q-Value", f"{self.test_organism.average_q_value:.3f}")
        ]

    def generate_simulation_stats(self):
        stats = [
            ("Organism Count", str(self.num_organisms)),
            ("Item Count", str(self.num_items)),
            ("Deceased Organisms", str(self.deceased_organisms)),
        ]
        
        # Add organism stats
        stats.extend(self.all_organism_stats)
        
        return stats
    
    @staticmethod
    def format_parameter_name(name):
        if name is None:
            return "None"
        return ' '.join(word.capitalize() for word in str(name).split('_'))

    def cleanup(self):
        if hasattr(self, 'performance_updater'):
            self.performance_updater.stop()
            self.performance_updater.join()
        if hasattr(self, 'ui_updater'):
            self.ui_updater.stop()
            self.ui_updater.wait()

    def __del__(self):
        self.cleanup()


class PerformanceUpdater(Thread):
    def __init__(self, simulation_state):
        super().__init__()
        self.simulation_state = simulation_state
        self.queue = Queue()
        self.running = True

    def run(self):
        while self.running:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            utilization = nvmlDeviceGetUtilizationRates(self.simulation_state.gpu_handle)
            gpu_usage = utilization.gpu
            learning_backlog = sum(organism.RL_algorithm.get_learning_backlog() 
                                   for organism in self.simulation_state.sim_engine.organisms)
            
            self.queue.put((cpu_usage, memory_usage, gpu_usage, learning_backlog))
            time.sleep(0.25)  # Update every 250ms

    def stop(self):
        self.running = False

class UIUpdater(QThread):
    update_signal = Signal(list, list, list, list, list)  # Added an extra list for simulation stats

    def __init__(self, simulation_state):
        super().__init__()
        self.simulation_state = simulation_state
        self.running = True
        self.current_section = 0
        self.update_interval = 0.1  # Update every 100ms

    def run(self):
        while self.running:
            if self.current_section == 0:
                organism_stats = self.simulation_state.generate_organism_stats()
                self.update_signal.emit(organism_stats, None, None, None, None)
            elif self.current_section == 1:
                performance_stats = self.simulation_state.generate_performance_stats()
                self.update_signal.emit(None, performance_stats, None, None, None)
            elif self.current_section == 2:
                training_stats = self.simulation_state.generate_training_stats()
                action_distribution = self.simulation_state.action_distribution
                self.update_signal.emit(None, None, training_stats, action_distribution, None)
            else:  # self.current_section == 3
                simulation_stats = self.simulation_state.generate_simulation_stats()
                self.update_signal.emit(None, None, None, None, simulation_stats)

            # Move to the next section
            self.current_section = (self.current_section + 1) % 4

            # Sleep for the update interval
            self.msleep(int(self.update_interval * 1000))

    def stop(self):
        self.running = False
