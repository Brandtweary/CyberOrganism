from collections import defaultdict

class SummaryLogger:
    def __init__(self):
        self.logs = defaultdict(list)

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

    def get_summary(self):
        summary = []
        for level in ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']:
            if self.logs[level]:
                summary.append(f"\n--- {level} Messages ---")
                summary.extend(self.logs[level])
        return '\n'.join(summary)

# Create an instance of the SummaryLogger
summary_logger = SummaryLogger()