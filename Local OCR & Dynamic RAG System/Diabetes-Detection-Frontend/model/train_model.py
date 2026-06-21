import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import joblib
import os




df=pd.read_csv('diabetes.csv')

#print(df.info())

# print(df.describe())

diabetes_data_copy = df.copy(deep = True)

diabetes_data_copy[['Glucose',
                    'BloodPressure',
                    'SkinThickness',
                    'Insulin',
                    'BMI']] = diabetes_data_copy[['Glucose',
                                                  'BloodPressure',
                                                  'SkinThickness',
                                                  'Insulin',
                                                  'BMI']].replace(0,np.nan)

# print(diabetes_data_copy.isnull().sum())

diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)

df_out = diabetes_data_copy
# print(diabetes_data_copy.isnull().sum())

# print(diabetes_data_copy.describe())
# print(df.describe())

# # Calculate the IQR for each numerical column
# Q1 = diabetes_data_copy.quantile(0.25)
# Q3 = diabetes_data_copy.quantile(0.75)
# IQR = Q3 - Q1

# print("---Q1--- \n",Q1)
# print("\n---Q3--- \n",Q3)
# print("\n---IQR---\n",IQR)

# # Define the upper and lower bounds for outliers
# lower_bound = Q1 - 5 * IQR
# upper_bound = Q3 + 5 * IQR

# # Create a boolean mask to identify outliers
# outlier_mask = (diabetes_data_copy < lower_bound) | (diabetes_data_copy > upper_bound)

# # Remove outliers
# df_out = diabetes_data_copy[~outlier_mask.any(axis=1)]

def load_and_preprocess():
    X = df_out.drop(columns=['Outcome'])
    y = df_out['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True  )

    scaler = StandardScaler()
    train_X = scaler.fit_transform(X_train)
    test_X = scaler.transform(X_test)

    return train_X, test_X, y_train, y_test, scaler

def train_and_evaluate(train_X, test_X, y_train, y_test, scaler, save_model=False):
    """
    Trains multiple classifiers, evaluates them, and saves the best model.
    Returns metrics dictionary and best model object.
    """

    # Define classifiers
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "RandomForest": RandomForestClassifier(n_estimators=1000, random_state=42),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=11)
    }

    best_model = None
    best_f1 = 0
    metrics_dict = {}

    for name, model in models.items():
        # Train
        model.fit(train_X, y_train)
        y_pred = model.predict(test_X)

        # Evaluate
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }
        metrics_dict[name] = metrics

        # Track best model (based on F1-score)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = model
    y_pred_best = best_model.predict(test_X)
    report = classification_report(y_test, y_pred_best, target_names=["Not Diabetic", "Diabetic"])
    print("\nClassification Report for Best Model:\n")
    print(report)

    # Save the best model and scaler
    if save_model:
        joblib.dump(best_model, os.path.join("model", "diabetes_model.joblib"))
        joblib.dump(scaler, os.path.join("model", "scaler.joblib"))
        joblib.dump((test_X, y_test), os.path.join("model", "test_data.joblib"))
        
    return metrics_dict, best_model


# Preprocess
train_X, test_X, y_train, y_test, scaler = load_and_preprocess()

# Train & Evaluate
metrics, best_model = train_and_evaluate(train_X, test_X, y_train, y_test, scaler, save_model=True)

print(metrics)
# print("Scaled Training Data:")
# display(X_train_scaled.head())

# print("\nScaled Test Data:")
# display(X_test_scaled.head())   