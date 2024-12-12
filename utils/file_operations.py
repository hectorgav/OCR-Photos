from typing import List
import os
import logging


def get_image_files(directory: str) -> List[str]:
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif')
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]
    if not image_files:
        logging.warning(f"No image files found in {directory}")
    return image_files
