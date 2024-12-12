# Stable Version with excellent results
import os
import logging
from tqdm import tqdm
import easyocr
from PIL import Image
import re
import shutil
import cv2
import numpy as np
from typing import Dict, List, Tuple

# Load configuration
from config.config import SOURCE_DIR, JOB_DIR, MAX_DISTANCE, BASE_NUMBER_MIN, BASE_NUMBER_MAX, SUFFIX_MAX, SUFFIX_MIN
from utils.setup_logging import setup_logging


def preprocess_image(image_path: str) -> Image.Image:
    """Preprocess the image for OCR."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    return dilation


def read_job_number_from_image(image_path: str, reader: easyocr.Reader) -> str:
    """Read job number from image using easyOCR."""
    try:
        img = preprocess_image(image_path)
        result = reader.readtext(np.array(img), allowlist='0123456789-:Job')
        text = ' '.join([entry[1] for entry in result])

        patterns = [
            r'(?:Job:|^)\s*(\d{6}-\d{2})',
            r'\b\d{6}-\d{2}\b',
            r'(\d{6}[-\s]\d{2})'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                job_number = matches[0].replace(' ', '')
                logging.info(f"Job number found in {image_path}: {job_number}")
                return job_number

        logging.warning(f"No job number found in {image_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading job number from {image_path}: {str(e)}")
        return None


def validate_job_number(job_number: str) -> bool:
    """Validate job number format and range."""
    if not job_number:
        return False
    pattern = r'^\d{6}-\d{2}$'
    if not re.match(pattern, job_number):
        return False
    base_number, suffix = map(int, job_number.split('-'))
    return BASE_NUMBER_MIN <= base_number <= BASE_NUMBER_MAX and SUFFIX_MIN <= suffix <= SUFFIX_MAX


def process_photos() -> Tuple[List[str], Dict[str, List[str]]]:
    """Process photos and organize them by job number."""
    reader = easyocr.Reader(['en'], gpu=False)
    file_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(file_extensions)])
    job_photos = {}
    current_job = None
    current_group = []

    with tqdm(total=len(image_files), desc="Processing images", unit="image",
              initial=1, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for image_file in image_files:
            image_path = os.path.join(SOURCE_DIR, image_file)
            job_number = read_job_number_from_image(image_path, reader)

            if job_number and validate_job_number(job_number):
                if current_job and job_number != current_job:
                    job_photos[current_job] = current_group
                    current_group = []
                current_job = job_number

            if current_job:
                current_group.append(image_file)
            else:
                logging.warning(f"No job number detected for {image_file}")

            pbar.update(1)

        # Add the last group
        if current_job:
            job_photos[current_job] = current_group

    # Move photos to their respective folders
    for job_number, photos in job_photos.items():
        job_folder = os.path.join(JOB_DIR, job_number)
        try:
            os.makedirs(job_folder, exist_ok=True)
            for photo in photos:
                source_path = os.path.join(SOURCE_DIR, photo)
                destination_path = os.path.join(job_folder, photo)
                shutil.move(source_path, destination_path)
            logging.info(f"Processed {len(photos)} photos for job number {job_number}")
        except Exception as e:
            logging.error(f"Error processing job number {job_number}: {str(e)}")

    logging.info("Photo processing completed.")
    return image_files, job_photos


def validate_grouping(image_files: List[str], job_photos: Dict[str, List[str]], max_distance: int) -> Dict[str, List[dict]]:
    """Validate photo groupings and identify suspicious gaps."""
    validation_results = {}
    for job_number, photos in job_photos.items():
        results = []
        try:
            photo_indices = sorted([image_files.index(p) for p in photos])
            for i in range(len(photo_indices) - 1):
                gap = photo_indices[i + 1] - photo_indices[i]
                if gap > max_distance:
                    results.append({
                        'gap_start': photos[i],
                        'gap_end': photos[i + 1],
                        'gap_size': gap,
                        'is_suspicious': True
                    })
                    logging.warning(f"Suspicious gap in job {job_number}: {gap} photos between {photos[i]} and {photos[i + 1]}")
            validation_results[job_number] = results
        except Exception as e:
            logging.error(f"Error validating job {job_number}: {str(e)}")
            validation_results[job_number] = [{'error': str(e), 'is_suspicious': True}]
    return validation_results


if __name__ == "__main__":
    setup_logging()
    try:
        image_files, job_photos = process_photos()
        validation_results = validate_grouping(image_files, job_photos, MAX_DISTANCE)
        # Process validation results as needed
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
