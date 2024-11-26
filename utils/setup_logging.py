import logging
import os
# custom modules
from config.config import LOG_DIR, LOG_FILE

def setup_logging():
    """Set up logging configuration."""
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        filename= os.path.join(LOG_DIR, LOG_FILE),
        level= logging.INFO,
        format= '%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
