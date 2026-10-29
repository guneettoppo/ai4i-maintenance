#!/usr/bin/env python3
"""AI4I 2020 Predictive Maintenance ML Project
Dataset: ai4i2020.csv (machine failure prediction with TWF/HDF/PWF/OSF/RNF)
Reference: Explainable_Artificial_Intelligence_for_Predictive_Maintenance_Applications.pdf (XAI approach)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# Load
print("Loading ai4i2020.csv...")
df = pd.read_csv(r"D:/ML PROJECT/ai4i2020.csv")
print(f"Shape: {df.shape}")
print(df['Machine failure'].value_counts())

# Features / target
X = df.drop(columns=['UDI','Product ID','Type','Machine failure','TWF','HDF','PWF','OSF','RNF'])
# Predict binary failure
X = df[['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]',
        'Torque [Nm]','Tool wear [min]']]
y = df['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model
clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# Predict
pred = clf.predict(X_test)
prob = clf.predict_proba(X_test)[:,1]
print("ROC-AUC:", roc_auc_score(y_test, prob))
print(classification_report(y_test, pred))

# Save model + predictions
joblib.dump(clf, r"D:/ML PROJECT/project/src/model.pkl")
pd.DataFrame({"actual": y_test, "predicted": pred, "prob_failure": prob}).to_csv(r"D:/ML PROJECT/project/outputs/predictions.csv", index=False)
print("Done. Model saved to src/model.pkl, predictions to outputs/predictions.csv")
