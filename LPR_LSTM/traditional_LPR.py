import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
# Specify Tesseract-OCR installation path (windows OS only)
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


def extract_license_plate(image_path):
    # read image
    image = cv2.imread(image_path)
    # Check if the image is loading correctly
    if image is None:
        print("Error: Unable to load image at path:", image_path)
        return

    # If the image is loaded successfully, proceed with subsequent processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Show grayscale image
    #plt.figure(figsize=(10, 6))
    #plt.subplot(131), plt.imshow(gray, cmap='gray'), plt.title('Gray Image')

    # Horizontal and vertical edge detection using Sobel operator
    sobel_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)  # horizontal direction
    sobel_y = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3)  # vertical direction

    # Combine horizontal and vertical edge detection results
    sobel_combined = cv2.bitwise_or(sobel_x, sobel_y)

    # Display Sobel edge detection results
    #plt.subplot(132), plt.imshow(sobel_combined, cmap='gray'), plt.title('Sobel Edge Detection')

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # Binarization
    #ret, threshold = cv2.threshold(sobel_combined, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # Display the binarization result
    #plt.subplot(133), plt.imshow(threshold, cmap='gray'), plt.title('Thresholding Result')
    #plt.show()

    # Erode and dilate images
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 6))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 2))

    dilation = cv2.dilate(threshold, element2, iterations=1)
    erosion = cv2.erode(dilation, element1, iterations=1)
    #dilation2 = cv2.dilate(erosion, element2, iterations=3)

    # Display expansion corrosion results
    #plt.figure(figsize=(10, 6))
    #plt.subplot(131), plt.imshow(dilation, cmap='gray'), plt.title('Dilation Result')
    #plt.subplot(132), plt.imshow(erosion, cmap='gray'), plt.title('Erosion Result')
    #plt.subplot(133), plt.imshow(dilation2, cmap='gray'), plt.title('Final Dilation Result')
    #plt.show()

    # Find license plate area
    license_plate = None
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 2 < aspect_ratio < 5.5:  # Normally the aspect ratio of a license plate is between 2 and 5.5
            license_plate = image[y:y + h, x:x + w]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break  # Assume there is only one license plate, exit the loop after finding it

    # Show results
    #plt.figure(figsize=(10, 6))
    #plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    #if license_plate is not None:
        #plt.subplot(122), plt.imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB)), plt.title('License Plate')
    #else:
        #plt.subplot(122), plt.imshow(np.zeros((50, 150, 3), dtype=np.uint8)), plt.title('License Plate Not Found')
    #plt.show()

    # Returns the recognized license plate image
    return license_plate

def read_license_plate(license_plate_img):
    if license_plate_img is not None:
        # Convert OpenCV image to PIL image format
        license_plate_img_pil = Image.fromarray(cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2RGB))

        # Tesseract configuration: use whitelist to limit character set
        config = '-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8'

        # OCR using Tesseract
        text = pytesseract.image_to_string(license_plate_img_pil, config=config)

        # Optional: Use regular expressions for post-processing
        # import re
        # pattern = re.compile("[A-Z0-9]+")
        # matches = pattern.findall(text)
        # if matches:
        #     text = " ".join(matches)

        #print(text)
        return text
    else:
        print("No license plate found")




if __name__ == "__main__":
    
    image_path = 'D:/archive/test/1/80-ZL-029.png'#for test
    license_plate_img = extract_license_plate(image_path)
    read_license_plate(license_plate_img)

