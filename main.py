import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
)
import joblib
import matplotlib.pyplot as plt

# 1️⃣ Load dataset
df = pd.read_csv("dataset.csv")
print("✅ Data loaded successfully!\n")
print(df.head())

# 2️⃣ Check for missing data
print("\nMissing values per column:\n", df.isnull().sum())

# 3️⃣ Split features and target
X = df.drop("price_range", axis=1)
y = df["price_range"]

# 4️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6️⃣ Train Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# 7️⃣ Evaluate model
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("\n✅ Model Evaluation:")
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8️⃣ Save model
joblib.dump(model, "mobile_price_model.pkl")
print("\n💾 Model saved as mobile_price_model.pkl")

# 9️⃣ Feature importance visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind="barh", title="Top 10 Features")
plt.tight_layout()
plt.show()
