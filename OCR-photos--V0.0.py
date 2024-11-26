import easyocr
import os
import re

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Specify the folder path containing the images
image_folder = 'images/'

# Define the regex pattern for job numbers
job_pattern = r'(?:Job:\s*)?(\d{6}-\d{2})'

# Function to extract job numbers from text
def extract_job_numbers(text):
    return re.findall(job_pattern, text, re.IGNORECASE)

# Iterate through all files in the specified folder
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        image_path = os.path.join(image_folder, filename)

        # Perform text recognition on the image
        result = reader.readtext(image_path, detail=0)

        # Join all detected text into a single string
        full_text = ' '.join(result)

        # Extract job numbers from the text
        job_numbers = extract_job_numbers(full_text)

        # Print the filename and extracted job numbers
        if job_numbers:
            print(f"File: {filename}")
            for job_number in job_numbers:
                print(f"Job Number: {job_number}")
            print("---")
