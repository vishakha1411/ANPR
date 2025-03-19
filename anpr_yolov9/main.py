import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import pytesseract
from datetime import datetime
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO('license_plate_detector.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('a.mp4')

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Region of interest (ROI)
area = [(27, 270), (16, 486), (1015, 481), (992, 280)]

count = 0
list1 = []
processed_numbers = set()

# Open file for writing car plate data
with open("car_plate_data.txt", "a") as file:
    file.write("NumberPlate\tDate\tTime\n")  # Writing column headers


def preprocess_for_ocr(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Bilateral Filter for smoothing while preserving edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Try adaptive thresholding to improve contrast
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Resize image for better OCR performance
    resized = cv2.resize(thresh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    return resized


def validate_plate(text):
    # Use regex to filter valid license plate formats (adjust to your region)
    # Example: License plates are typically alphanumeric
    return re.match(r"^[A-Z0-9]{5,10}$", text)


def run_ocr_on_plate(crop):
    preprocessed = preprocess_for_ocr(crop)

    # Configure Tesseract for alphanumeric-only OCR and page segmentation
    config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    # Run OCR on the preprocessed image
    text = pytesseract.image_to_string(preprocessed, config=config).strip()
    text = re.sub(r'[^A-Z0-9]', '', text)  # Remove unwanted characters

    # Filter by confidence (Optional step if Tesseract provides confidence)
    conf = pytesseract.image_to_data(preprocessed, config=config, output_type=pytesseract.Output.DICT)['conf']
    if int(conf[0]) < 60:  # Example confidence threshold
        return None

    # Return the cleaned-up text
    return text
while True:
    ret, frame = cap.read()
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])

        d = int(row[5])
        c = class_list[d]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
        if result >= 0:
            crop = frame[y1:y2, x1:x2]

            # Preprocess the cropped image for better OCR
            preprocessed = preprocess_for_ocr(crop)

            # Perform OCR
            text = pytesseract.image_to_string(preprocessed, config='--psm 8').strip()
            text = re.sub(r'[^A-Za-z0-9]', '', text)  # Remove unwanted characters

            # Only process valid license plates
            if validate_plate(text) and text not in processed_numbers:
                processed_numbers.add(text)
                list1.append(text)
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with open("car_plate_data.txt", "a") as file:
                    file.write(f"{text}\t{current_datetime}\n")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.imshow('crop', crop)

    # Draw the ROI polygon
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
