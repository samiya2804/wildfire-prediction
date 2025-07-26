import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# 1. Load data
df = pd.read_csv("forestfires.csv")
df['fire'] = df['area'].apply(lambda x: 1 if x > 0 else 0)

# 2. Drop unused columns
df = df.drop(['month', 'day', 'area'], axis=1)
df.head()

# Features and label
X = df.drop('fire', axis=1)
y = df['fire']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "wildfire_model.pkl")