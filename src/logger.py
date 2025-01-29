import logging
import os
from pathlib import Path

def setup_logger():
    """Setup debug logger that writes to logs/debug.log"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Clear existing log file
    log_file = log_dir / "debug.log"
    if log_file.exists():
        log_file.unlink()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
        ]
    )
    
    return logging.getLogger("cyberorganism")
