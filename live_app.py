import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions
import pickle
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time

MODEL_PATH = 'hand_landmarker.task'
PKL_MODEL_PATH = 'model.pkl'

# מילון תרגום מאנגלית לעברית
HEBREW_TRANSLATION = {
    'ALEF': 'א', 'BEIT': 'ב', 'GIMEL': 'ג', 'DALED': 'ד', 'HEI': 'ה',
    'VAV': 'ו', 'ZAIN': 'ז', 'CHET': 'ח', 'TET': 'ט', 'YOOD': 'י',
    'CAF': 'כ', 'LAMED': 'ל', 'MEM': 'מ', 'NOON': 'נ', 'SAMECH': 'ס',
    'AIN': 'ע', 'PEI': 'פ', 'ZADIK': 'צ', 'KOOF': 'ק', 'REISH': 'ר',
    'SHIN': 'ש', 'TAF': 'ת', 'SPACE': ' '
}

with open(PKL_MODEL_PATH, 'rb') as f:
    clf_model = pickle.load(f)

options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

try:
    font = ImageFont.truetype("arial.ttf", 26)
    large_font = ImageFont.truetype("arial.ttf", 45)
except IOError:
    font = ImageFont.load_default()
    large_font = ImageFont.load_default()

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
    (15,16),(13,17),(17,18),(18,19),(19,20),(0,17)
]

# משתני חוויית משתמש (UX) והיסטוריה
current_sentence = ""
last_saved_sentence = "None"  
last_predicted_label = None
label_stable_start_time = None
REQUIRED_STABLE_TIME = 1.2  
confidence_pct = 0  

cap = cv2.VideoCapture(0)
print("האפליקציה פועלת! לחיצה על S (או ד') תשמור את המשפט מיד.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    panel = np.zeros((220, frame.shape[1], 3), dtype=np.uint8)
    frame = np.vstack((frame, panel))

    rgb_frame = cv2.cvtColor(frame[:480, :], cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    detected_text = "?"
    confidence_pct = 0

    if results.hand_landmarks:
        full_features = []
        for idx in range(2):
            if idx < len(results.hand_landmarks):
                lm = results.hand_landmarks[idx]
                full_features.extend([val for p in lm for val in (p.x, p.y, p.z)])

                for connection in HAND_CONNECTIONS:
                    a, b = connection
                    ax, ay = int(lm[a].x * frame.shape[1]), int(lm[a].y * frame.shape[0])
                    bx, by = int(lm[b].x * frame.shape[1]), int(lm[b].y * frame.shape[0])
                    cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 2)
            else:
                full_features.extend([0.0] * 63)
        
        predicted_label = clf_model.predict([full_features])[0]
        probabilities = clf_model.predict_proba([full_features])[0]
        max_prob_idx = np.argmax(probabilities)
        confidence_pct = int(probabilities[max_prob_idx] * 100)

        hebrew_letter = HEBREW_TRANSLATION.get(predicted_label, predicted_label)
        detected_text = hebrew_letter

        if predicted_label == last_predicted_label:
            if label_stable_start_time is not None:
                if time.time() - label_stable_start_time >= REQUIRED_STABLE_TIME:
                    current_sentence += hebrew_letter
                    label_stable_start_time = None 
                    last_predicted_label = None
        else:
            last_predicted_label = predicted_label
            label_stable_start_time = time.time()
    else:
        last_predicted_label = None
        label_stable_start_time = None

    # --- עיצוב גרפי וציור ה-UI ---
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    draw.text((30, 490), f"Confidence: {confidence_pct}%", font=font, fill=(255, 255, 255))
    draw.rectangle([250, 495, 450, 515], fill=(50, 50, 50))
    draw.rectangle([250, 495, 250 + (confidence_pct * 2), 515], fill=(0, 120, 255))

    draw.text((30, 530), "Current Sign: ", font=font, fill=(0, 255, 0))
    draw.text((180, 530), get_display(detected_text), font=font, fill=(0, 255, 0))
    
    draw.text((30, 570), "Text: ", font=large_font, fill=(255, 255, 0))
    draw.text((140, 570), get_display(current_sentence), font=large_font, fill=(255, 255, 0))
    
    draw.text((30, 640), "History: ", font=font, fill=(180, 180, 180))
    draw.text((140, 640), get_display(last_saved_sentence), font=font, fill=(180, 180, 180))

    # עדכון המקרא על המסך שיהיה ברור
    instructions = "[SPACE] - Space | [X/ס] - Delete | [S/ד] - Save | [C/ב] - Clear | [Q] - Quit"
    draw.text((20, 15), instructions, font=font, fill=(255, 255, 255))

    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow('ASL Hebrew Real-Time Translator', frame)
    
    # --- טיפול בכפתורי המקלדת (תומך גם בעברית וגם באנגלית!) ---
    key = cv2.waitKey(1) & 0xFF
    
    # בדיקת המקשים הלחוצים (כולל המקבילה שלהם בעברית במקלדת)
    if key == ord('q') or key == ord('ת'):  
        break
    elif key == 32:  # מקש SPACE במקלדת
        current_sentence += " "
    elif key == ord('x') or key == ord('ס') or key == 8:  # מחיקת אות אחרונה (X או ס' או Backspace)
        current_sentence = current_sentence[:-1]
    elif key == ord('c') or key == ord('ב'):  # נקה הכל (C או ב')
        current_sentence = ""
    elif key == ord('s') or key == ord('ד'):  # שמירה (S או ד') - בלי קונטרול!
        if current_sentence.strip():
            last_saved_sentence = current_sentence  
            with open("exported_sentences.txt", "a", encoding="utf-8") as txt_file:
                txt_file.write(current_sentence + "\n")
            print(f"Saved: {current_sentence}")
            current_sentence = ""  

cap.release()
cv2.destroyAllWindows()
detector.close()