import os
import logging
from tqdm import tqdm
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
import shutil
from pdf2image import convert_from_path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2


SOURCE_DIR = r"C:\Photos-repo\testing"
MAX_WORKERS = 4  # Adjust based on your system's capabilities


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


logger = setup_logging()


def preprocess_image(image):
    gray = image.convert('L')
    enhancer = ImageEnhance.Contrast(gray)
    contrast = enhancer.enhance(2)
    sharpened = contrast.filter(ImageFilter.SHARPEN)
    return sharpened.point(lambda x: 0 if x < 128 else 255, '1')


def read_job_number_from_image(image_path):
    try:
        image = convert_from_path(image_path)[0] if image_path.lower().endswith('.pdf') else Image.open(image_path)
        preprocessed = preprocess_image(image)

        job_numbers = []
        for psm in [3, 4, 6]:
            text = pytesseract.image_to_string(preprocessed, config=f'--psm {psm}')
            job_numbers.extend(re.findall(r'\d{6}-\d{2}', text))

        if job_numbers:
            job_number = max(set(job_numbers), key=job_numbers.count)
            logger.info(f"Job number found in {image_path}: {job_number}")
            return job_number
        else:
            logger.warning(f"No job number found in {image_path}")
    except Exception as e:
        logger.error(f"Error reading job number from {image_path}: {str(e)}")
    return None


def process_image(image_file):
    image_path = os.path.join(SOURCE_DIR, image_file)
    return image_file, read_job_number_from_image(image_path)


def move_photos(job_photos):
    for job_number, photos in job_photos.items():
        job_folder = os.path.join(SOURCE_DIR, job_number)
        try:
            os.makedirs(job_folder, exist_ok=True)
            logger.info(f"Created folder: {job_folder}")

            for photo in photos:
                source_path = os.path.join(SOURCE_DIR, photo)
                destination_path = os.path.join(job_folder, photo)
                shutil.move(source_path, destination_path)
                logger.info(f"Moved {photo} to {job_folder}")

            logger.info(f"Processed {len(photos)} photos for job number {job_number}")
        except Exception as e:
            logger.error(f"Error processing job number {job_number}: {str(e)}")


def process_photos():
    image_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.jpeg'))])
    job_photos = defaultdict(list)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_image = {executor.submit(process_image, image_file): image_file for image_file in image_files}

        for future in tqdm(as_completed(future_to_image), total=len(image_files), desc="Processing images", unit="image"):
            image_file, job_number = future.result()
            if job_number:
                job_photos[job_number].append(image_file)

    move_photos(job_photos)
    logger.info("Photo processing completed.")


if __name__ == "__main__":
    try:
        process_photos()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")