import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("--- טעינת הנתונים מהקובץ שלך ---")
df = pd.read_csv('dataset.csv')

print("כמות הדגימות שקיימות מכל אות בטבלה:")
print(df['label'].value_counts())

# הפרדת הנתונים
X = df.drop('label', axis=1)
y = df['label']

print("\n--- אימון מודל Random Forest ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# בדיקת הדיוק
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'💥 אחוז הדיוק של המודל שלך: {accuracy:.2%}')

print("\n--- דוח סיווג מפורט (Classification Report) ---")
print(classification_report(y_test, y_pred))

print("\n--- מטריצת בלבול (Confusion Matrix) ---")
# מציג טבלה שמראה מול מה כל אות התבלבלה
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
print(cm_df)

# שמירת המודל
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n🏆 המודל ומטריצת הבלבול נשמרו בהצלחה!")