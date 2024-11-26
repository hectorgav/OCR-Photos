import os
import re
import shutil
import logging
from typing import List, Dict, Tuple
from tqdm import tqdm
import cv2
import easyocr
# custom modules
from config.config import SOURCE_DIR, JOB_DIR
from utils.setup_logging import setup_logging

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)
# reader = easyocr.Reader(['en'], contrast_ths=0.1, adjust_contrast=0.5)


def read_and_sort_images(directory: str) -> List[str]:
    """Read and sort image files from the given directory."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]
    return sorted(images, key=lambda x: os.path.getmtime(os.path.join(directory, x)))


def extract_job_number(image_path: str) -> str:
    """Extract job number from image using OCR with improved preprocessing and regex."""
    try:
        # Read and preprocess the image
        image = cv2.imread(image_path)

        # Perform OCR on the adjusted image
        result = reader.readtext(image, detail=0)
        text = ' '.join(result)

        # Multiple regex pattern for more robust job number extraction
        patterns = [
            r'\b\d{6}\s*-\s*\d{2}\b',
            r'(?:Job:\s*)?(\d{6}-\d{2})'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0).replace(' ', '')

        return ''
    except Exception as e:
        logging.error(f"OCR error for {image_path}: {str(e)}")
        return ''


def validate_job_number(job_number: str) -> bool:
    """Validate the format of a job number."""
    pattern = r'^\d{6}-\d{2}$'
    return bool(re.match(pattern, job_number))


def group_photos_by_job_numbers(images: List[str]) -> Dict[str, List[str]]:
    """Group photos by their extracted job numbers."""
    groups = {}
    for image in tqdm(images, desc="Grouping photos"):
        job_number = extract_job_number(os.path.join(SOURCE_DIR, image))
        if validate_job_number(job_number):
            groups.setdefault(job_number, []).append(image)
        else:
            logging.warning(f"Invalid or missing job number for {image}")
    return groups


def organize_photos_by_job(groups: Dict[str, List[str]]) -> None:
    """Organize photos into job folders, appending to existing folders if they exist."""
    for job_number, photos in groups.items():
        job_folder = os.path.join(JOB_DIR, job_number)
        os.makedirs(job_folder, exist_ok=True)
        for photo in photos:
            try:
                source_path = os.path.join(SOURCE_DIR, photo)
                destination_path = os.path.join(job_folder, photo)
                if os.path.exists(destination_path):
                    # If the file already exists, append a number to the filename
                    base, extension = os.path.splitext(photo)
                    counter = 1
                    while os.path.exists(destination_path):
                        new_filename = f"{base}_{counter}{extension}"
                        destination_path = os.path.join(job_folder, new_filename)
                        counter += 1
                shutil.move(source_path, destination_path)
            except Exception as e:
                logging.error(f"Error moving {photo} to {job_folder}: {str(e)}")


def validate_photo_groupings(groups: Dict[str, List[str]]) -> List[str]:
    """Validate photo groupings and identify potential issues."""
    warnings = []
    for job_number, photos in groups.items():
        if len(photos) < 2:
            warnings.append(f"Job {job_number} has only {len(photos)} photo(s)")
        # Add more validation checks as needed
    return warnings


def summarize_processing_results(groups: Dict[str, List[str]], warnings: List[str]) -> str:
    """Generate a summary report of processing results."""
    total_photos = sum(len(photos) for photos in groups.values())
    summary = f"Processed {total_photos} photos into {len(groups)} job folders.\n"
    if warnings:
        summary += f"{len(warnings)} warnings were generated:\n"
        summary += "\n".join(warnings)
    return summary


def process_photos_with_progress() -> Tuple[Dict[str, List[str]], List[str]]:
    """Process photos with progress tracking."""
    images = read_and_sort_images(SOURCE_DIR)
    groups = group_photos_by_job_numbers(images)
    organize_photos_by_job(groups)
    warnings = validate_photo_groupings(groups)
    return groups, warnings


def main():
    # Set up logging
    setup_logging()

    logging.info("Starting photo processing")
    try:
        groups, warnings = process_photos_with_progress()
        summary = summarize_processing_results(groups, warnings)
        logging.info(summary)
    except Exception as e:
        logging.error(f"An error occurred during processing: {str(e)}")
    logging.info("Photo processing completed")

if __name__ == "__main__":
    main()