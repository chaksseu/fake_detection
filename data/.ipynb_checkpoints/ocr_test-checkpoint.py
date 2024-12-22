import cv2
import easyocr
import os

def detect_and_save_digits_easyocr(image_path, output_folder):
    """
    Detect digits using EasyOCR and save bounding boxes as images.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Perform OCR
    results = reader.readtext(image, detail=1)

    for i, (bbox, text, confidence) in enumerate(results):
        if confidence < 0.5 or not text.isdigit():  # Filter low-confidence and non-digit results
            continue

        # Extract bounding box coordinates
        (top_left, top_right, bottom_right, bottom_left) = bbox
        x_min = int(min(top_left[0], bottom_left[0]))
        y_min = int(min(top_left[1], top_right[1]))
        x_max = int(max(bottom_right[0], top_right[0]))
        y_max = int(max(bottom_right[1], bottom_left[1]))

        # Crop and save the digit
        digit = image[y_min:y_max, x_min:x_max]
        output_path = os.path.join(output_folder, f'digit_{i}_{text}.png')
        cv2.imwrite(output_path, digit)
        print(f"Saved digit '{text}' at {output_path}")

# Example usage
image_path = 'data/augmented/train/normal/normal_scan_6/pos/pos_original.png' 
output_folder = "output_easyocr"
detect_and_save_digits_easyocr(image_path, output_folder)
