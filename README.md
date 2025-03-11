# pythelpers

A collection of Python helper functions and utilities for common tasks.

## Installation

```bash
pip install pythelpers
```

Or install from source:

```bash
git clone https://github.com/yourusername/pythelpers.git
cd pythelpers
pip install -e .
```

## Features

### Logging Utilities

#### Standard Logger Usage

```python
from pythelpers.logger import setup_logger

# Create a logger that logs to both console and file
logger = setup_logger(
    name="my_app",
    log_file="application.log",
    log_level="INFO",
    log_dir="logs"
)

logger.info("Application started")
logger.warning("Something might be wrong")
logger.error("An error occurred")
```

#### Capturing Print Statements

```python
from pythelpers.logger import log_to_file

# Use with a context manager to log all print statements
with log_to_file(title="Data Processing"):
    print("Starting data processing...")
    # Your code here
    print("Processing complete!")
```

### File Utilities

```python
from pythelpers.utils import ensure_directory_exists, get_timestamp_filename, safe_filename ▋