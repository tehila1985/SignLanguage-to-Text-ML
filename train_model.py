import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

print("--- שלב 3: טעינת הנתונים ובדיקה ---")
# טעינת קובץ הנתונים שאספת
df = pd.read_csv('dataset.csv')

# בדיקה אילו אותיות קיימות בקובץ וכמה דגימות יש מכל אות
print("התפלגות האותיות שנאספו ב-Dataset שלך:")
print(df['label'].value_counts())
print("-" * 40)

# הפרדת הנתונים: X מכיל את 126 המספרים של היד, y מכיל את האות (התווית)
X = df.drop('label', axis=1)
y = df['label']

print("--- שלב 4: פיצול הנתונים ואימון מודל ---")
# פיצול הנתונים: 80% ללמידה ו-20% לבדיקת איכות המודל
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# יצירת מודל Random Forest ואימון שלו על הנתונים שלך
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# בדיקת אחוז הדיוק של המודל
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'💥 אחוז הדיוק של המודל שלך (Accuracy): {accuracy:.2%}')

print("-" * 40)
print("--- שלב 6: שמירת המודל המאומן ---")
# שמירת המודל לקובץ חיצוני כדי שנוכל להשתמש בו באפליקציה החיה
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("🏆 הקובץ model.pkl נשמר בהצלחה בתיקייה! מוכן לשימוש חיו.")