import os
import time
from traditional_LPR import extract_license_plate, read_license_plate

# path setting
image_folder = 'D:/588way2/archive/test'  # Replace with picture folder path
max_images_to_process = 1000  # Set the maximum number of images to process

# statistical variables
total_images = 0
correctly_identified = 0
error_cases = []  # Used to record recognition errors

# Recording time starts
start_time = time.time()

# Get all image file names and limit the number of processes
image_files = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
image_files = image_files[:max_images_to_process]  # Limit the number of processes

# Iterate through all image files in the folder
for image_name in image_files:
    total_images += 1
    image_path = os.path.join(image_folder, image_name)

    # Locate and identify license plates
    license_plate_image = extract_license_plate(image_path)
    identified_text = read_license_plate(license_plate_image)

    # Extract correct answer from file name
    correct_answer = image_name.split('.')[0].replace('-', '')

    # Compare results and handle possible errors
    if identified_text:
        identified_text_cleaned = identified_text.strip().replace(" ", "").upper()
        if identified_text_cleaned == correct_answer.upper():
            correctly_identified += 1
        else:
            error_cases.append((correct_answer, identified_text_cleaned))
    else:
        error_cases.append((correct_answer, 'None'))

# Print error cases, one line per case
for case in error_cases:
    print(f'Correct: {case[0]}, Identified: {case[1]}')

# Calculate running time
end_time = time.time()
total_time = end_time - start_time

# Calculate accuracy
accuracy = (correctly_identified / total_images) * 100

# Output statistical results
#print(error_cases)
print(f'Total images processed: {total_images}')
print(f'Correctly identified: {correctly_identified}')
print(f'Accuracy: {accuracy:.2f}%')
print(f'Total runtime: {total_time:.2f} seconds')
