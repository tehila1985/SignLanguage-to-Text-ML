import kagglehub
import ssl

# עקיפת חסימות אינטרנט אפשריות בהורדה
ssl._create_default_https_context = ssl._create_unverified_context

print("מתחיל להוריד את תיקיית התמונות של המורה, זה עשוי לקחת כמה דקות...")
path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print("הורדה הסתיימה בהצלחה! התמונות שמורות בכתובת הבאה:")
print(path)