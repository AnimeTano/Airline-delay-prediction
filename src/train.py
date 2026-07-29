import joblib 
import os

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix


X_train = pd.read_csv("./data/processed/X_train.csv")
X_val = pd.read_csv("./data/processed/X_val.csv")
y_train = pd.read_csv("./data/processed/y_train.csv").values.ravel()
y_val = pd.read_csv("./data/processed/y_val.csv").values.ravel()

categorical_columns = ['carrier', 'airport', 'season', 'carrier_name', 'airport_name']
numerical_columns = ['arr_flights', 'arr_cancelled', 'arr_diverted', 'month_sin', 'month_cos', 'arr_del15_lag1',
       'arr_del15_lag3', 'arr_del15_lag6', 'arr_delay_lag1', 'arr_delay_lag3', 'arr_delay_lag6']

encoder = TargetEncoder(
    cols = categorical_columns,
    smoothing = 1.0,
    handle_missing = 'value',
    handle_unknown = 'value'
)

X_train_enc = X_train.copy()
X_val_enc = X_val.copy()

X_train_enc = encoder.fit_transform(X_train, y_train).fillna(0)
X_val_enc = encoder.transform(X_val).fillna(0)

scaler = StandardScaler()

X_train_scaled = X_train_enc.copy()
X_val_scaled = X_val_enc.copy()
X_train_scaled = scaler.fit_transform(X_train_scaled)
X_val_scaled = scaler.transform(X_val_scaled)

# Бейзлайн(Логистическая регрессия)
lg = LogisticRegression(
    random_state = 42,
      max_iter = 1500
)
lg.fit(X_train_scaled, y_train)

# Оценка
pred_lg = lg.predict_proba(X_val_scaled)[:, 1]
auc_lg = roc_auc_score(y_val, pred_lg)

print(f"Validation ROC-AUC: {auc_lg:.4f}")

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

pred_rf = rf.predict_proba(X_val_scaled)[:, 1]
auc_rf = roc_auc_score(y_val, pred_rf)

print(f"Random Forest ROC-AUC: {auc_rf:.4f}")

# XGBoost
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb.fit(X_train_scaled, y_train)

pred_xgb = xgb.predict_proba(X_val_scaled)[:, 1]
auc_xgb = roc_auc_score(y_val, pred_xgb)

print(f"XGBoost ROC-AUC: {auc_xgb:.4f}")

pred = 0.01 * pred_lg + 0.09 * pred_rf + 0.9 * pred_xgb
auc_all = roc_auc_score(y_val, pred)
print(f'Ensemble ROC-AUC: {auc_all:.4f}')

# ----------------------------------------------------------
os.makedirs('models', exist_ok = True)
joblib.dump(xgb, 'models/xgb_model.pk1')
joblib.dump(encoder, 'models/target_encoder.pk1')
joblib.dump(scaler, 'models/scaler.pk1')