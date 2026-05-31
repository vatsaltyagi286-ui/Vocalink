import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import pyttsx3
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# ---------------- INIT ----------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")  # 🔥 changed theme

engine = pyttsx3.init()

cap = None
camera_running = False

detector = HandDetector(maxHands=1)

classifier = Classifier(
    "C:/Users/amans/OneDrive/Desktop/HAND SIGN/Model/keras_model.h5",
    "C:/Users/amans/OneDrive/Desktop/HAND SIGN/Model/labels.txt"
)

offset = 20
imgSize = 300

labels = ["Bathroom & Restroom","Call & Number","Dislike", "Drink",  "Goodbye",
          "GoodLuck","Heart", "HII", "Hungry","I",  "I love you","Looser",
          "NO", "Okay","Peace"
          , "Please","Question","Sorry","Stand", "Thanks", "Water","YES",]

last_prediction = ""
confidence_threshold = 0.85

# ---------------- UI ----------------
app = ctk.CTk()
app.title(" Smart Sign Language Translator")
app.geometry("1200x650")

# Gradient-style title
title = ctk.CTkLabel(
    app,
    text="🤟 AI Sign Language Translator",
    font=ctk.CTkFont(size=28, weight="bold"),
    text_color="#00FFCC"
)
title.pack(pady=15)

main_frame = ctk.CTkFrame(app, corner_radius=15)
main_frame.pack(fill="both", expand=True, padx=20, pady=10)

left_frame = ctk.CTkFrame(main_frame, corner_radius=15)
left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

right_frame = ctk.CTkFrame(main_frame, corner_radius=15)
right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

# Camera display
camera_label = ctk.CTkLabel(left_frame, text="")
camera_label.pack(fill="both", expand=True, padx=10, pady=10)

# Output text
output_text = ctk.CTkTextbox(
    right_frame,
    font=("Arial", 24, "bold"),
    corner_radius=10,
    fg_color="#1a1a1a",
    text_color="#00FFAA"
)
output_text.pack(fill="both", expand=True, padx=10, pady=10)

# Status + Confidence
status_label = ctk.CTkLabel(right_frame, text="Camera: OFF ", text_color="red")
status_label.pack(pady=5)

confidence_label = ctk.CTkLabel(right_frame, text="Confidence: 0%", text_color="yellow")
confidence_label.pack(pady=5)

# ---------------- FUNCTIONS ----------------
def speak_text():
    text = output_text.get("1.0", "end").strip()
    if text:
        engine.say(text)
        engine.runAndWait()

def clear_text():
    output_text.delete("1.0", "end")

def start_camera():
    global cap, camera_running
    if not camera_running:
        cap = cv2.VideoCapture(0)
        camera_running = True
        status_label.configure(text="Camera: ON ", text_color="green")
        update_camera()

def stop_camera():
    global cap, camera_running
    camera_running = False
    status_label.configure(text="Camera: OFF ", text_color="red")
    if cap:
        cap.release()
    camera_label.configure(image=None)

def update_camera():
    global last_prediction

    if not camera_running:
        return

    success, img = cap.read()
    if not success:
        app.after(10, update_camera)
        return

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

        if imgCrop.size != 0:
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
            except:
                pass

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            confidence = max(prediction)

            confidence_label.configure(
                text=f"Confidence: {int(confidence * 100)}%"
            )

            if confidence > confidence_threshold:
                current_label = labels[index]

                if current_label != last_prediction:
                    output_text.delete("1.0", "end")
                    output_text.insert("end", current_label)
                    last_prediction = current_label

            cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (0, 255, 200), 3)

    rgb = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    camera_label.configure(image=imgtk)
    camera_label.image = imgtk

    app.after(10, update_camera)

# ---------------- BUTTONS ----------------
btn_frame = ctk.CTkFrame(right_frame)
btn_frame.pack(pady=10)

def styled_button(parent, text, command, color):
    return ctk.CTkButton(
        parent,
        text=text,
        command=command,
        corner_radius=20,
        fg_color=color,
        hover_color="#333333",
        font=("Arial", 14, "bold")
    )

start_btn = styled_button(btn_frame, "Start ", start_camera, "#00AA55")
start_btn.grid(row=0, column=0, padx=10)

stop_btn = styled_button(btn_frame, "Stop ", stop_camera, "#CC3333")
stop_btn.grid(row=0, column=1, padx=10)

speak_btn = styled_button(btn_frame, "Speak ", speak_text, "#3366FF")
speak_btn.grid(row=0, column=2, padx=10)

clear_btn = styled_button(btn_frame, "Clear ", clear_text, "#FF8800")
clear_btn.grid(row=0, column=3, padx=10)

# ---------------- RUN ----------------
app.mainloop()

if cap:
    cap.release()
cv2.destroyAllWindows()