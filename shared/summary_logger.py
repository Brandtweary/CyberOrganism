from collections import defaultdict
import statistics
from threading import Lock


class SummaryLogger:
    def __init__(self):
        self.log_levels = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
        self.active_log_levels = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
        self.new_logs = defaultdict(list)
        self.all_logs = defaultdict(list)
        self.new_organism_metrics = defaultdict(lambda: defaultdict(list))
        self.all_organism_metrics = defaultdict(lambda: defaultdict(list))
        self.new_organism_logs = defaultdict(lambda: defaultdict(list))
        self.all_organism_logs = defaultdict(lambda: defaultdict(list))
        self.organism_metrics_lock = Lock()
        self.logs_lock = Lock()
        self.organism_logging = True
        self.frame_logging = True
        self.periodic_logging = True
        self.final_logging = True
        self.organism_metrics_sample = True
        self.test_organism_id = ''

    def log(self, level, message):
        if level not in self.active_log_levels:
            return
        with self.logs_lock:
            self.new_logs[level].append(message)
            self.all_logs[level].append(message)

    def debug(self, message):
        self.log('DEBUG', message)

    def info(self, message):
        self.log('INFO', message)

    def warning(self, message):
        self.log('WARNING', message)

    def error(self, message):
        self.log('ERROR', message)

    def critical(self, message):
        self.log('CRITICAL', message)

    def log_organism_metrics(self, organism_id: str, metrics: dict):
        if not isinstance(organism_id, str):
            raise ValueError("Organism ID must be a string.")
        if not self.organism_logging:
            return
        with self.organism_metrics_lock:
            for key, value in metrics.items():
                self.new_organism_metrics[organism_id][key].append(value)
                self.all_organism_metrics[organism_id][key].append(value)

    def add_organism_log(self, organism_id: str, logs: dict):
        if not isinstance(organism_id, str):
            raise ValueError("Organism ID must be a string.")
        
        with self.logs_lock:
            for level, messages in logs.items():
                if level == 'METRICS':
                    # Handle metrics separately
                    for metric_name, metric_values in messages.items():
                        for value in metric_values:
                            # Create a single-item dict for each metric value
                            metric_dict = {metric_name: value}
                            self.log_organism_metrics(organism_id, metric_dict)
                elif level in self.active_log_levels:
                    self.new_organism_logs[organism_id][level].extend(messages)
                    self.all_organism_logs[organism_id][level].extend(messages)

    def get_new_log_summary(self):
        summary = []
        for level in self.active_log_levels:
            if level in self.new_logs and self.new_logs[level]:
                summary.append(f"\n--- {level} Messages ---")
                summary.extend(self.new_logs[level])
                self.new_logs[level].clear()
        
        if self.new_organism_logs:
            summary.append("\n--- Organism Logs ---")
            for organism_id, logs in self.new_organism_logs.items():
                summary.append(f"\nOrganism {organism_id}:")
                for level in self.active_log_levels:
                    if level in logs and logs[level]:
                        summary.append(f"  {level}:")
                        summary.extend(f"    {msg}" for msg in logs[level])
            self.new_organism_logs.clear()
        
        if self.new_organism_metrics:
            summary.append("\n--- Organism Metrics ---")
            test_organism_id = self.test_organism_id
            
            for metric in self.new_organism_metrics[test_organism_id]:
                test_values = self.new_organism_metrics[test_organism_id][metric]
                all_values = [self.new_organism_metrics[org][metric] for org in self.new_organism_metrics.keys()]
                all_values = [item for sublist in all_values for item in sublist]  # Flatten

                if all(isinstance(v, (int, float)) for v in test_values + all_values):
                    test_max = max(test_values) if test_values else 0
                    test_avg = statistics.mean(test_values) if test_values else 0
                    all_max = max(max(self.new_organism_metrics[org][metric]) if self.new_organism_metrics[org][metric] else 0 
                                   for org in self.new_organism_metrics.keys())
                    all_avg = statistics.mean(all_values) if all_values else 0

                    summary.append(f"{metric}:")
                    summary.append(f"  Test organism max: {test_max:.4f}")
                    summary.append(f"  Average organism max: {all_max:.4f}")
                    summary.append(f"  Difference in max: {test_max - all_max:.4f}")
                    summary.append(f"  Test organism avg: {test_avg:.4f}")
                    summary.append(f"  All organisms avg: {all_avg:.4f}")
                    summary.append(f"  Difference in avg: {test_avg - all_avg:.4f}")
                else:
                    test_value = test_values[-1] if test_values else "N/A"
                    all_values_unique = set(all_values)

                    summary.append(f"{metric}:")
                    summary.append(f"  Test organism: {test_value}")
                    summary.append(f"  All organisms: {', '.join(map(str, all_values_unique))}")

            self.new_organism_metrics.clear()

        return '\n'.join(summary)

    def get_overall_log_summary(self):
        summary = []
        for level in self.active_log_levels:
            if self.all_logs[level]:
                summary.append(f"\n--- {level} Messages ---")
                summary.extend(self.all_logs[level])
        
        if self.all_organism_logs:
            summary.append("\n--- Overall Organism Logs ---")
            for organism_id, logs in self.all_organism_logs.items():
                summary.append(f"\nOrganism {organism_id}:")
                for level in self.active_log_levels:
                    if level in logs and logs[level]:
                        summary.append(f"  {level}:")
                        summary.extend(f"    {msg}" for msg in logs[level])
        
        if self.all_organism_metrics:
            summary.append("\n--- Overall Organism Metrics ---")
            for metric in next(iter(self.all_organism_metrics.values())):
                all_values = [self.all_organism_metrics[org][metric] for org in self.all_organism_metrics.keys()]
                all_values = [item for sublist in all_values for item in sublist]  # Flatten

                if all(isinstance(v, (int, float)) for v in all_values):
                    max_value = max(all_values) if all_values else 0
                    avg_value = statistics.mean(all_values) if all_values else 0
                    summary.append(f"{metric}:")
                    summary.append(f"  Max: {max_value:.4f}")
                    summary.append(f"  Avg: {avg_value:.4f}")
                else:
                    unique_values = set(all_values)
                    summary.append(f"{metric}: {', '.join(map(str, unique_values))}")

        return '\n'.join(summary)

    def get_organism_metrics_sample(self):
        if not self.organism_metrics_sample or not self.all_organism_metrics:
            return ""

        summary = ["\n--- Organism Metrics Sample ---"]
        test_organism_id = self.test_organism_id
        comparison_organism_id = next((org_id for org_id in self.all_organism_metrics.keys() if org_id != test_organism_id), None)
        
        for metric in self.all_organism_metrics[test_organism_id]:
            summary.append(f"\n{metric}:")
            test_values = self.all_organism_metrics[test_organism_id][metric][:5]
            summary.append(f"  Test organism {test_organism_id} (first 5 values): {test_values}")
            
            if comparison_organism_id:
                comparison_values = self.all_organism_metrics[comparison_organism_id][metric][:5]
                summary.append(f"  Comparison organism {comparison_organism_id} (first 5 values): {comparison_values}")

        return '\n'.join(summary)

    def get_frame_log_summary(self):
        if not self.frame_logging:
            return ""
        return self.get_new_log_summary()

    def get_periodic_log_summary(self):
        if not self.periodic_logging or self.frame_logging:  # periodic logs are redundant if logging every frame
            return ""
        return self.get_new_log_summary()

    def get_final_log_summary(self):
        if not self.final_logging:
            return ""
        summary = self.get_overall_log_summary()
        if self.organism_metrics_sample:
            summary += self.get_organism_metrics_sample()
        return summary

    def set_test_organism_id(self, organism_id):
        self.test_organism_id = organism_id

# Create and export a single instance of SummaryLogger
summary_logger = SummaryLogger()
