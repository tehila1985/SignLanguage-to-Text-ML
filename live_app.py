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

# מילון תרגום
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
    font = ImageFont.truetype("arial.ttf", 35)
    large_font = ImageFont.truetype("arial.ttf", 55)
except IOError:
    font = ImageFont.load_default()
    large_font = ImageFont.load_default()

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
    (15,16),(13,17),(17,18),(18,19),(19,20),(0,17)
]

# משתנים לצבירת המשפט
current_sentence = ""
last_predicted_label = None
label_stable_start_time = None
REQUIRED_STABLE_TIME = 1.0  # זמן בשניות שצריך להחזיק את היד יציבה כדי שהאות תוקלד

cap = cv2.VideoCapture(0)
print("האפליקציה החיה עם צבירת משפטים התחילה! (q ליציאה, c למחיקת משפט)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    detected_text = "?"
    
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
        
        # זיהוי האות הנוכחית
        predicted_label = clf_model.predict([full_features])[0]
        hebrew_letter = HEBREW_TRANSLATION.get(predicted_label, predicted_label)
        detected_text = hebrew_letter

        # מנגנון טיימר לצבירת האותיות (Debounce)
        if predicted_label == last_predicted_label:
            if label_stable_start_time is not None:
                elapsed = time.time() - label_stable_start_time
                # אם עברה שנייה והאות עדיין יציבה
                if elapsed >= REQUIRED_STABLE_TIME:
                    current_sentence += hebrew_letter
                    # מאפסים את הטיימר כדי שלא יקליד שוב בלולאה בלי להחליף אות
                    label_stable_start_time = None 
                    last_predicted_label = None
        else:
            last_predicted_label = predicted_label
            label_stable_start_time = time.time()
    else:
        # אם אין יד על המסך, מאפסים את הטיימר
        last_predicted_label = None
        label_stable_start_time = None

    # הכנת הטקסטים לתצוגה בעברית תקינה (RTL)
    display_current = f"אות נוכחית: {get_display(detected_text)}"
    display_sentence = f"משפט: {get_display(current_sentence)}"

    # ציור בעזרת Pillow
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # הצגת האות הנוכחית והמשפט המצטבר
    draw.text((50, 30), display_current, font=font, fill=(255, 0, 0))
    draw.text((50, 80), display_sentence, font=large_font, fill=(255, 255, 0))
    
    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow('ASL Hebrew Real-Time Translator', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        current_sentence = ""  # מחיקת המשפט בלחיצה על c

cap.release()
cv2.destroyAllWindows()
detector.close()