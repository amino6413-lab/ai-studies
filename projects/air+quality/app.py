#!/usr/bin/env python
# coding: utf-8

# =====================
# 1. 데이터 로드
# =====================
from ucimlrepo import fetch_ucirepo

air_quality = fetch_ucirepo(id=360)

X = air_quality.data.features
y = air_quality.data.targets

print(air_quality.metadata)
print(air_quality.variables)


# =====================
# 2. 데이터프레임 생성
# =====================
import pandas as pd

df = pd.concat([X, y], axis=1)

print(df.shape)
print(df.head())


# =====================
# 3. 결측치 변환
# =====================
# -200 값을 결측치로 변환
df.replace(-200, pd.NA, inplace=True)

print(df.isnull().sum())


# =====================
# 4. 컬럼 정리 + 결측치 처리
# =====================
# NMHC(GT) 컬럼 제거 (결측치 너무 많음)
df.drop(columns=['NMHC(GT)', 'Date', 'Time'], inplace=True)

# 나머지 결측치는 평균으로 채우기
df.fillna(df.mean(), inplace=True)

print(df.isnull().sum())
print(df.shape)


# =====================
# 5. 학습/테스트 분할 + Random Forest
# =====================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 목표변수: CO(GT) 예측
X = df.drop(columns=['CO(GT)'])
y = df['CO(GT)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print(f"Random Forest R2 Score: {r2_score(y_test, y_pred_rf):.4f}")
print(f"Random Forest RMSE: {mean_squared_error(y_test, y_pred_rf)**.5:.4f}")


# =====================
# 6. XGBoost 기본 학습
# =====================
from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
print(f"XGBoost R2 Score: {r2_score(y_test, y_pred_xgb):.4f}")
print(f"XGBoost RMSE: {mean_squared_error(y_test, y_pred_xgb)**.5:.4f}")


# =====================
# 7. XGBoost 하이퍼파라미터 튜닝 (GridSearchCV)
# =====================
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xgb_tune = XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'alpha': [0, 1],
    'lambda': [1, 10]
}

grid = GridSearchCV(
    estimator=xgb_tune,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)
print("Best Parameters:", grid.best_params_)

# 최적 모델로 예측 및 평가
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
print(f"Tuned XGBoost R2 Score: {r2_score(y_test, y_pred_best):.4f}")
print(f"Tuned XGBoost RMSE: {mean_squared_error(y_test, y_pred_best)**.5:.4f}")


# =====================
# 8. 시각화 (3개 모델 비교)
# =====================
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, y_pred, title, color in zip(
    axes,
    [y_pred_rf, y_pred_xgb, y_pred_best],
    ['Random Forest: 실제 vs 예측', 'XGBoost: 실제 vs 예측', 'Tuned XGBoost: 실제 vs 예측'],
    ['blue', 'green', 'orange']
):
    ax.scatter(y_test, y_pred, alpha=0.3, color=color)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_title(title)
    ax.set_xlabel('실제값')
    ax.set_ylabel('예측값')

plt.tight_layout()
plt.show()


# =====================
# 9. Feature 중요도
# =====================
importances = rf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names, orient='h')
plt.title('Random Forest - Feature 중요도')
plt.xlabel('중요도')
plt.tight_layout()
plt.show()
