
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# noinspection PyTypeChecker

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

classifier = Classifier("C:/Users/amans/OneDrive/Desktop/HAND SIGN/Model/keras_model.h5", "C:/Users/amans/OneDrive/Desktop/HAND SIGN/Model/abels.txt")

offset = 20
imgSize = 300

labels = ["Bathroom & Restroom","Call & Number","Dislike", "Drink",  "Goodbye",
          "GoodLuck","Heart", "HII", "Hungry","I",  "I love you","Looser",
          "NO", "Okay","Peace"
          , "Please","Question","Sorry","Stand", "Thanks", "Water","YES",]

while True:
    success, img = cap.read()

    if not success:
        print("Camera not detected!")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(img.shape[1], x + w + offset)
        y2 = min(img.shape[0], y + h + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        crop_h, crop_w, _ = imgCrop.shape
        aspectRatio = crop_h / crop_w

        try:
            if aspectRatio > 1:
                k = imgSize / crop_h
                wCal = math.ceil(k * crop_w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / crop_w
                hCal = math.ceil(k * crop_h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hGap + hCal, :] = imgResize

        except Exception:
            continue

        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x1, y1 - 50), (x1 + 100, y1),
                      (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x1 + 5, y1 - 15),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)

        cv2.rectangle(imgOutput, (x1, y1), (x2, y2),
                      (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)