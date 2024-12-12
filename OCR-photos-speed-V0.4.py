import os
import logging
from tqdm import tqdm
import easyocr
import re
import shutil
import cv2
import numpy as np
from typing import Dict, List
import cProfile

# Load configuration
from config.config import SOURCE_DIR, JOB_DIR, MAX_DISTANCE, BASE_NUMBER_MIN, BASE_NUMBER_MAX, SUFFIX_MAX, SUFFIX_MIN
from utils.setup_logging import setup_logging
from utils.file_operations import get_image_files


def preprocess_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    return dilation


def read_job_number_from_image(image_path: str, reader: easyocr.Reader) -> tuple[bool, str]:
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
                logging.debug(f"Detected job number before validation: {job_number}")
                return True, job_number
        logging.warning(f"No job number found in {image_path}")
        return False, ''
    except Exception as e:
        logging.error(f"Error reading job number from {image_path}: {str(e)}")
        return False, ''


def validate_job_number(job_number: str) -> bool:
    if not job_number:
        return False
    pattern = r'^\d{6}-\d{2}$'
    if not re.match(pattern, job_number):
        return False
    base_number, suffix = map(int, job_number.split('-'))
    return BASE_NUMBER_MIN <= base_number <= BASE_NUMBER_MAX and SUFFIX_MIN <= suffix <= SUFFIX_MAX


def validate_grouping(image_files: List[str], job_photos: Dict[str, List[str]], max_distance: int) -> Dict[
    str, List[dict]]:
    validation_results = {}
    for job_number, photos in list(job_photos.items()):
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
                    logging.warning(
                        f"Suspicious gap in job {job_number}: {gap} photos between {photos[i]} and {photos[i + 1]}")
            validation_results[job_number] = results
        except Exception as e:
            logging.error(f"Error validating job {job_number}: {str(e)}")
            validation_results[job_number] = [{'error': str(e), 'is_suspicious': True}]
    return validation_results


def move_photos_to_folders(job_photos: Dict[str, List[str]], source_dir: str, job_dir: str) -> None:
    for job_number, photos in list(job_photos.items()):
        job_folder = os.path.join(job_dir, job_number)
        try:
            os.makedirs(job_folder, exist_ok=True)
            for photo in photos:
                source_path = os.path.join(source_dir, photo)
                destination_path = os.path.join(job_folder, photo)
                shutil.move(source_path, destination_path)
            logging.info(f"Processed {len(photos)} photos for job number {job_number}")
        except Exception as e:
            logging.error(f"Error processing job number {job_number}: {str(e)}")


def post_process_unrecognized_photos(image_files: List[str], job_photos: Dict[str, List[str]]) -> Dict[str, List[str]]:
    index_to_job = {}
    for job_number, photos in list(job_photos.items()):
        for photo in photos:
            idx = image_files.index(photo)
            index_to_job[idx] = job_number
    sorted_indices = sorted(list(index_to_job.keys()))

    for i, image_file in enumerate(image_files):
        if any(image_file in photos for photos in list(job_photos.values())):
            continue
        prev_idx = next((idx for idx in reversed(sorted_indices) if idx < i), None)
        next_idx = next((idx for idx in sorted_indices if idx > i), None)
        if prev_idx is not None and next_idx is not None:
            prev_job = index_to_job[prev_idx]
            next_job = index_to_job[next_idx]
            assign_job = prev_job if prev_job == next_job else prev_job
            if assign_job not in job_photos:
                job_photos[assign_job] = []
            job_photos[assign_job].append(image_file)
            logging.info(f"Assigned {image_file} to job {assign_job} based on sequence")
    return job_photos


def process_photos(image_files: List[str], reader: easyocr.Reader) -> Dict[str, List[str]]:
    """Process photos and organize them by job number."""
    job_photos = {}

    with tqdm(image_files, desc="Processing images", unit="image",
              initial=1, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for image_file in pbar:
            image_path = os.path.join(SOURCE_DIR, image_file)
            success, job_number = read_job_number_from_image(image_path, reader)
            if success and validate_job_number(job_number):
                if job_number not in job_photos:
                    job_photos[job_number] = []
                job_photos[job_number].append(os.path.basename(image_file))
                logging.info(f"Job number found in {image_path}: {job_number}")
            else:
                logging.warning(f"No valid job number detected for {image_file}")
            pbar.set_postfix({"Current": os.path.basename(image_file)})

    # Post-process unrecognized photos
    logging.info("Post-processing unrecognized photos...")
    job_photos = post_process_unrecognized_photos(image_files, job_photos)

    # Validate grouping
    logging.info("Validating photo groupings...")
    validation_results = validate_grouping(image_files, job_photos, MAX_DISTANCE)
    logging.info(f"Validation results: {validation_results}")

    # Move files to respective folders
    logging.info("Moving photos to job folders...")
    move_photos_to_folders(job_photos, SOURCE_DIR, JOB_DIR)

    logging.info("Photo processing completed.")
    return job_photos



if __name__ == "__main__":
    setup_logging()
    try:
        image_files = get_image_files(SOURCE_DIR)
        if not image_files:
            raise ValueError("No image files found in source directory")

        reader = easyocr.Reader(['en'], gpu=False)
        job_photos = process_photos(image_files, reader)

        if job_photos:
            validation_results = validate_grouping(image_files, job_photos, MAX_DISTANCE)
            logging.info(f"Validation results: {validation_results}")
        else:
            logging.warning("No jobs were processed")

    except ValueError as ve:
        logging.error(f"ValueError occurred: {str(ve)}")
    except TypeError as te:
        logging.error(f"TypeError occurred: {str(te)}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        logging.error(f"Error occurred in function: {__name__}")
    finally:
        logging.info("Script execution completed.")
