




import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

folder = "Data/Water"
counter = 0

# Create folder if not exists
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    if not success:
        print("Camera not detected!")
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # BOUNDARY FIX
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(img.shape[1], x + w + offset)
        y2 = min(img.shape[0], y + h + offset)

        # Crop safely
        imgCrop = img[y1:y2, x1:x2]

        #  VERY IMPORTANT: skip if imgCrop is empty
        if imgCrop.size == 0:
            print("Skipped empty crop frame.")
            cv2.imshow("Image", img)
            cv2.waitKey(1)
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCropShape = imgCrop.shape

        # If crop is too small, skip it
        if imgCropShape[0] < 10 or imgCropShape[1] < 10:
            print("Crop too small, skipping...")
            continue

        aspectRatio = imgCropShape[0] / imgCropShape[1]

        if aspectRatio > 1:
            k = imgSize / imgCropShape[0]
            wCal = math.ceil(k * imgCropShape[1])

            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize

        else:
            k = imgSize / imgCropShape[1]
            hCal = math.ceil(k * imgCropShape[0])

            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print("Saved:", counter)
