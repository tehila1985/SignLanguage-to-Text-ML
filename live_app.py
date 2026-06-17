import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions
import pickle
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont  # ספריות לציור פונטים תומכי עברית
import numpy as np

MODEL_PATH = 'hand_landmarker.task'
PKL_MODEL_PATH = 'model.pkl'

# מילון תרגום מהשמות ב-CSV לאותיות בעברית
HEBREW_TRANSLATION = {
    'ALEF': 'א', 'BEIT': 'ב', 'GIMEL': 'ג', 'DALED': 'ד', 'HEI': 'ה',
    'VAV': 'ו', 'ZAIN': 'ז', 'CHET': 'ח', 'TET': 'ט', 'YOOD': 'י',
    'CAF': 'כ', 'LAMED': 'ל', 'MEM': 'מ', 'NOON': 'נ', 'SAMECH': 'ס',
    'AIN': 'ע', 'PEI': 'פ', 'ZADIK': 'צ', 'KOOF': 'ק', 'REISH': 'ר',
    'SHIN': 'ש', 'TAF': 'ת'
}

# טעינת המודל שאימנת
with open(PKL_MODEL_PATH, 'rb') as f:
    clf_model = pickle.load(f)

# הגדרת MediaPipe Tasks ל-2 ידיים
options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# טעינת פונט Arial המובנה של ווינדוס לתמיכה בעברית
try:
    font = ImageFont.truetype("arial.ttf", 45)
except IOError:
    font = ImageFont.load_default()

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
    (15,16),(13,17),(17,18),(18,19),(19,20),(0,17)
]

cap = cv2.VideoCapture(0)
print("האפליקציה החיה בעברית מתוקנת מתחילה! (לחיצה על q ליציאה)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        full_features = []
        
        for idx in range(2):
            if idx < len(results.hand_landmarks):
                lm = results.hand_landmarks[idx]
                full_features.extend([val for p in lm for val in (p.x, p.y, p.z)])

                # ציור השלד הירוק
                for connection in HAND_CONNECTIONS:
                    a, b = connection
                    ax, ay = int(lm[a].x * frame.shape[1]), int(lm[a].y * frame.shape[0])
                    bx, by = int(lm[b].x * frame.shape[1]), int(lm[b].y * frame.shape[0])
                    cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 2)
            else:
                full_features.extend([0.0] * 63)
        
        # ניחוש האות על ידי המודל
        predicted_label = clf_model.predict([full_features])[0]
        hebrew_letter = HEBREW_TRANSLATION.get(predicted_label, predicted_label)
        
        # סידור כיוון העברית ושילוב בטקסט
        display_text = f"תוא: {get_display(hebrew_letter)}"

        # --- מנגנון הציור בעברית בעזרת PIL ---
        # המרת הפריים הנוכחי מ-OpenCV (תמונת BGR) לתמונת PIL (RGB)
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # כתיבת הטקסט בעברית על פני התמונה
        draw.text((50, 40), display_text, font=font, fill=(255, 0, 0))
        
        # החזרת התמונה בחזרה לפורמט ש-OpenCV מכיר
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow('ASL Hebrew Real-Time Translator', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()