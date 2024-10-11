from collections import defaultdict
import statistics
from threading import Lock


class SummaryLogger:
    def __init__(self):
        self.logs = defaultdict(list)
        self.organism_metrics = defaultdict(lambda: defaultdict(list))
        self.organism_metrics_lock = Lock()
        self.organism_logging = True

    def log(self, level, message):
        self.logs[level].append(message)

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

    def log_organism_metrics(self, organism_id, metrics):
        if not self.organism_logging:
            return
        with self.organism_metrics_lock:
            for key, value in metrics.items():
                self.organism_metrics[organism_id][key].append(value)

    def get_summary(self):
        summary = []
        for level in ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']:
            if self.logs[level]:
                summary.append(f"\n--- {level} Messages ---")
                summary.extend(self.logs[level])
        
        if not self.organism_metrics:
            summary.append("\n--- No Organism Metrics Recorded ---")
            return '\n'.join(summary)

        summary.append("\n--- Organism Metrics ---")
        
        test_organism_id = next(iter(self.organism_metrics.keys()), None)
        if test_organism_id is None:
            summary.append("No organisms recorded.")
            return '\n'.join(summary)

        comparison_organism_id = next((org_id for org_id in self.organism_metrics.keys() if org_id != test_organism_id), None)
        
        for metric in self.organism_metrics[test_organism_id]:
            summary.append(f"\n{metric}:")
            test_values = self.organism_metrics[test_organism_id][metric][:5]
            summary.append(f"  Test organism {test_organism_id} (first 5 values): {test_values}")
            
            if comparison_organism_id:
                comparison_values = self.organism_metrics[comparison_organism_id][metric][:5]
                summary.append(f"  Comparison organism {comparison_organism_id} (first 5 values): {comparison_values}")

        return '\n'.join(summary)

    def get_periodic_summary(self, test_organism_id):
        summary = []
        all_organisms = list(self.organism_metrics.keys())
        
        for metric in self.organism_metrics[test_organism_id]:
            test_values = self.organism_metrics[test_organism_id][metric]
            all_values = [self.organism_metrics[org][metric] for org in all_organisms]
            all_values = [item for sublist in all_values for item in sublist]  # Flatten

            if all(isinstance(v, (int, float)) for v in test_values + all_values):
                # Numeric values - calculate average and difference
                test_avg = statistics.mean(test_values) if test_values else 0
                all_avg = statistics.mean(all_values) if all_values else 0

                summary.append(f"{metric}:")
                summary.append(f"  Test organism: {test_avg:.4f}")
                summary.append(f"  All organisms: {all_avg:.4f}")
                summary.append(f"  Difference: {test_avg - all_avg:.4f}")
            else:
                # Non-numeric values - just print them plainly
                test_value = test_values[-1] if test_values else "N/A"
                all_values_unique = set(all_values)

                summary.append(f"{metric}:")
                summary.append(f"  Test organism: {test_value}")
                summary.append(f"  All organisms: {', '.join(map(str, all_values_unique))}")

        return '\n'.join(summary)

# Create an instance of the SummaryLogger
summary_logger = SummaryLogger()