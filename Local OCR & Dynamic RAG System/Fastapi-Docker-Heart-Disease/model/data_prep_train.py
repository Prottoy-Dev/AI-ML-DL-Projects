import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

DATA_PATH = "heart.csv"


def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    print("Data sample:\n", df.head())

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_and_evaluate(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler):
    # Logistic Regression on scaled data
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    y_pred_logreg = logreg.predict(X_test_scaled)
    acc_logreg = accuracy_score(y_test, y_pred_logreg)
    print(f"Logistic Regression accuracy: {acc_logreg:.4f}")

    # Random Forest on original (unscaled) data
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest accuracy: {acc_rf:.4f}")

    # Save models and scaler
    os.makedirs("model", exist_ok=True)
    joblib.dump(logreg, os.path.join("model", "logreg_model.joblib"))
    joblib.dump(rf, os.path.join("model", "rf_model.joblib"))
    joblib.dump(scaler, os.path.join("model", "scaler.joblib"))
    print("Models and scaler saved in 'model/' folder.")

if __name__ == "__main__":
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess()
    train_and_evaluate(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler)
