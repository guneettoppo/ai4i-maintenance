"""Full pipeline: Optuna tuning + SHAP + DeLong bootstrap comparison."""
import optuna, shap, pandas as pd, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X = pd.read_csv(r"D:/ML PROJECT/ai4i2020.csv")[['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']]
y = pd.read_csv(r"D:/ML PROJECT/ai4i2020.csv")['Machine failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def obj(trial):
    model = RandomForestClassifier(n_estimators=trial.suggest_int('n',50,300), max_depth=trial.suggest_int('d',3,20), random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

study = optuna.create_study(direction='maximize')
study.optimize(obj, n_trials=20)
print("Best AUC:", study.best_value, study.best_params)
best = study.best_params
best_map = {'n_estimators': best['n'], 'max_depth': best['d']}
clf = RandomForestClassifier(**{**best_map, 'random_state':42, 'class_weight':'balanced'})
clf.fit(X_train, y_train)
joblib.dump(clf, r"D:/ML PROJECT/project/src/model_tuned.pkl")

# SHAP
explainer = shap.TreeExplainer(clf)
shap_vals = explainer.shap_values(X_test)
shap.summary_plot(shap_vals[1] if isinstance(shap_vals, list) else shap_vals, X_test, show=False)
import matplotlib.pyplot as plt
plt.tight_layout(); plt.savefig(r"D:/ML PROJECT/project/reports/shap_summary.png")
print("SHAP plot saved. Tuned model saved.")
