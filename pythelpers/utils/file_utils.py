import os
import re
import datetime
from pathlib import Path


def ensure_directory_exists(directory_path):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory_path (str or Path): Path of directory to create
    
    Returns:
        Path: Path object for created directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp_filename(base_name, extension=None, timestamp_format="%Y-%m-%d_%H-%M-%S"):
    """
    Generate a filename with a timestamp.
    
    Args:
        base_name (str): Base name for the file
        extension (str, optional): File extension. Defaults to None.
        timestamp_format (str, optional): Format for the timestamp. Defaults to "%Y-%m-%d_%H-%M-%S".
    
    Returns:
        str: Filename with timestamp
    """
    timestamp = datetime.datetime.now().strftime(timestamp_format)
    if extension:
        if extension.startswith('.'):
            return f"{base_name}_{timestamp}{extension}"
        else:
            return f"{base_name}_{timestamp}.{extension}"
    return f"{base_name}_{timestamp}"


def safe_filename(filename):
    """
    Convert a string to a safe filename.
    
    Args:
        filename (str): Input string to convert
    
    Returns:
        str: Safe filename
    """
    # Remove invalid chars
    s = re.sub(r'[\\/*?:"<>|]', "", filename)
    # Replace spaces and multiple underscores with single underscore
    s = re.sub(r'[\s_]+', "_", s)
    # Remove leading/trailing underscores and periods
    return s.strip("_.")