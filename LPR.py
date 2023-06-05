import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import pandas as pd
import datetime as dt
from PIL import Image
import pytesseract

text_val_list = []
date_val_list = []
columns = ['Date/Time', 'Text']
flag = 0
text = ""

# open the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera or video stream
    ret, frame = cap.read()
    if not ret:
        break
    # convert the feed into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("preview", gray)
    # Noise reduction
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    # Edge detection
    edged = cv2.Canny(bfilter, 30, 200)

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is not None:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)
        plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Point tessaract_cmd to tessaract.exe

        # Open image with PIL
        img = Image.fromarray(cropped_image)

        # Extract text from image
        text = str(pytesseract.image_to_string(img))
        print(text)

        # enter date and time in excel
        now = dt.datetime.now()
        date_time = str(now.strftime("%m/%d/%Y %H:%M:%S"))
        date_val_list.append(date_time)
        text_val_list.append(text)
        if text != "" and len(text) > 8:
            break
            flag = 1
    if flag == 1:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(date_val_list)
print(text_val_list)
date_val_list = [date_val_list[len(date_val_list) - 1]]
text_val_list = [text_val_list[len(text_val_list) - 1]]

# create an excel file
df = pd.DataFrame(list(zip(date_val_list, text_val_list)), columns=columns)
df.to_excel("LPR.xlsx", index=False)

cap.release()
cv2.destroyAllWindows()
