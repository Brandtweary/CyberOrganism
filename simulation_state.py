import psutil
import time


class SimulationState:
    def __init__(self, simulation_engine, ui):
        self.sim_engine = simulation_engine
        self.ui = ui
        self.current_state = simulation_engine.current_state
        self.test_organism = simulation_engine.test_organism
        self.input_parameters = {param: getattr(self.test_organism, param) for param in self.test_organism.input_parameters}
        self.display_parameters = {param: getattr(self.test_organism, param) for param in self.test_organism.display_parameters}
        self.training_metrics = self.test_organism.training_metrics
        self.cpu_usage = psutil.cpu_percent()
        self.memory_usage = psutil.virtual_memory().percent
        self.available_memory = psutil.virtual_memory().available / (1024 * 1024)
        self.framerate = 0.0
            
        # Summary statistics
        self.time_low_loss = 0
        self.start_time = time.time()
        self.total_time = 0

        # Add training statistics
        self.training_stats = self.test_organism.RL_algorithm.training_stats.get_stats()
        self.training_record_stats = self.test_organism.RL_algorithm.training_stats.record_stats

        # Add new attributes for storing average times
        self.avg_update_times = []
        self.avg_draw_times = []
        self.avg_total_frame_times = []

    def generate_organism_stats(self):
        stats = []
        for param, value in self.display_parameters.items():
            formatted_name = self.format_parameter_name(param)
            formatted_value = f"{value:.3f}" if isinstance(value, float) else value
            stats.append(f"{formatted_name}: {formatted_value}")
        return stats

    def generate_performance_stats(self):
        stats = [
            f"CPU Usage: {self.cpu_usage:.1f}%",
            f"Memory Usage: {self.memory_usage:.1f}%",
            f"Available Memory: {self.available_memory:.1f} MB",
            f"FPS: {self.framerate:.1f}",
            f"Simulation Time: {int(self.total_time)} seconds",
        ]
        return stats
    
    def generate_training_stats(self):
        stats = []
        for stat_type, stat_info in self.training_record_stats.items():
            if stat_info['record']:
                stats.append(f"{stat_type}:")
                for stat_name in stat_info['stat_names']:
                    if stat_name in self.training_stats:
                        values = self.training_stats[stat_name]
                        if values:
                            value = values[-1]
                            formatted_value = f"{value:.3f}" if isinstance(value, float) else value
                            stats.append(f"  {stat_name}: {formatted_value}")
        return stats

    def update(self):        
        self.sim_engine.update_simulation()
        self.current_state = self.sim_engine.current_state
        self.input_parameters = {param: getattr(self.test_organism, param) for param in self.test_organism.input_parameters}
        self.display_parameters = {param: getattr(self.test_organism, param) for param in self.test_organism.display_parameters}
        self.training_metrics = self.test_organism.training_metrics
        
        # Update summary statistics
        current_loss = self.training_metrics['combined_averages']['loss_window_avg']
        if current_loss < 0.5:
            self.time_low_loss += self.sim_engine.UPDATE_INTERVAL
        self.total_time = time.time() - self.start_time
        
        # Update performance metrics
        self.cpu_usage = psutil.cpu_percent()
        self.memory_usage = psutil.virtual_memory().percent
        self.available_memory = psutil.virtual_memory().available / (1024 * 1024)

        # Update training statistics
        self.training_stats = self.test_organism.RL_algorithm.training_stats.get_stats()
        self.training_record_stats = self.test_organism.RL_algorithm.training_stats.record_stats

        # Update UI stats
        organism_stats = self.generate_organism_stats()
        performance_stats = self.generate_performance_stats()
        self.ui.update_left_sidebar(organism_stats, performance_stats, self.training_metrics)

    @staticmethod
    def format_parameter_name(name):
        if name is None:
            return "None"
        try:
            return ' '.join(word.capitalize() for word in str(name).split('_'))
        except AttributeError:
            return str(name)