import os
import sys
import logging
import datetime
from pathlib import Path
from contextlib import contextmanager


class TeeLogger:
    """
    A class that duplicates output to both the terminal and a log file.
    Useful for capturing print statements in scripts and notebooks.
    """
    def __init__(self, filename, logs_dirpath="logs"):
        """
        Initialize the TeeLogger with a filename and optional log directory.
        
        Args:
            filename (str): Name of the log file
            log_dir (str, optional): Directory to store log files. Defaults to "logs".
        """
        # Create logs directory if it doesn't exist
        Path(logs_dirpath).mkdir(exist_ok=True)
        
        # Create full path for log file
        self.log_path = logs_dirpath / filename
        self.terminal = sys.stdout
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        
        print(f"Logging to: {self.log_path}")

    def write(self, message):
        """Write message to both terminal and log file"""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        """Flush both terminal and log file"""
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        """Close the log file"""
        self.log_file.close()


@contextmanager
def file_only_output(logger):
    """Temporarily redirect output to file only"""
    original_terminal = logger.terminal
    logger.terminal = open(os.devnull, 'w')  # Null device
    try:
        yield
    finally:
        logger.terminal.close()
        logger.terminal = original_terminal

def _start_logging(title=None, descr=None, logs_dirpath="logs"):
    """Start logging all print outputs to a file with timestamp and optional title"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if title:
        # Clean title to be filesystem-friendly
        clean_title = ''.join(c if c.isalnum() else '_' for c in title).rstrip('_')
        filename = f"{timestamp}_{clean_title}.log"
    else:
        filename = f"{timestamp}.log"
    
    # Redirect stdout to our custom logger
    logger = TeeLogger(filename, logs_dirpath)
    original_stdout = sys.stdout
    sys.stdout = logger
    
    # Print a header in the log
    print(f"=== Log started at {timestamp} ===")
    if title:
        print(f"=== {title} ===")
    if descr:
        print(f"{descr}")
    print("")
    
    return logger, original_stdout

def start_logging(title=None, descr=None, logs_dirpath="logs"):
    """Start logging all print outputs to a file with timestamp and optional title"""
    logger, _ = _start_logging(title=title, descr=None, logs_dirpath=logs_dirpath)
    
    return logger


def stop_logging(logger):
    """Stop logging and restore normal print behavior"""
    sys.stdout = logger.terminal
    logger.close()
    print("Logging stopped")


@contextmanager
def start_logging2(title=None, logs_dirpath="logs"):
    """
    Context manager for logging print statements to a file.
    
    Args:
        title (str, optional): Title to include in the log file name. Defaults to None.
        log_dir (str, optional): Directory to store log files. Defaults to "logs".
    
    Yields:
        None: Use this in a with statement
    """
    logger, original_stdout = _start_logging(title=title, descr=None, logs_dirpath=logs_dirpath)
    
    try:
        yield
    finally:
        print("\n=== Log ended at {} ===".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ))
        sys.stdout = original_stdout
        logger.close()
        print(f"Logging completed. Log saved to {logger.log_path}")





def setup_logger(name, log_file=None, log_level=logging.INFO, log_format=None, 
                 log_dir="logs", timestamp=True, console=True):
    """
    Set up a logger with file and/or console output.
    
    Args:
        name (str): Name of the logger
        log_file (str, optional): Name of the log file. If None, no file logging. Defaults to None.
        log_level (int, optional): Logging level. Defaults to logging.INFO.
        log_format (str, optional): Log format string. Defaults to None (uses standard format).
        log_dir (str, optional): Directory for log files. Defaults to "logs".
        timestamp (bool, optional): Whether to add timestamp to log filename. Defaults to True.
        console (bool, optional): Whether to log to console. Defaults to True.
    
    Returns:
        Logger: Configured logger object
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers = []  # Clear any existing handlers
    
    # Define format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)
        
        # Add timestamp to filename if requested
        if timestamp:
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base, ext = os.path.splitext(log_file)
            log_file = f"{base}_{timestamp_str}{ext}"
        
        file_path = log_dir / log_file
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {file_path}")
    
    return logger


def get_logger(name):
    """
    Get a logger by name. If it doesn't exist, creates a basic one.
    
    Args:
        name (str): Name of the logger
    
    Returns:
        Logger: Logger object
    """
    logger = logging.getLogger(name)
    
    # If logger doesn't have handlers, add a basic one
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger