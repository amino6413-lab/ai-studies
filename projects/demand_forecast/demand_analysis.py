#!/usr/bin/env python
# coding: utf-8

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# =====================
# 1. 데이터 로드
# =====================
df = pd.read_csv('archive/Historical Product Demand.csv')
print("=" * 50)
print("1. 데이터 로드")
print("=" * 50)
print(df.shape)
print(df.head(10))
print(df.dtypes)


# =====================
# 2. 결측치 및 중복값 통계 (전처리 전)
# =====================
print("\n" + "=" * 50)
print("2. 전처리 전 - 결측치 통계")
print("=" * 50)
print(df.isnull().sum())
print(f"\n중복 행: {df.duplicated().sum()}")
print(f"데이터 크기: {df.shape[0]:,} 행, {df.shape[1]} 열")


# =====================
# 3. 전처리
# =====================
def clean_demand(val):
    val = str(val).strip()
    if val.startswith('(') and val.endswith(')'):
        try:
            return -float(val[1:-1].replace(',', ''))
        except:
            return np.nan
    try:
        return float(val.replace(',', ''))
    except:
        return np.nan

df['Order_Demand'] = df['Order_Demand'].apply(clean_demand)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date', 'Order_Demand'])
df = df[df['Order_Demand'] > 0]
df['YearMonth'] = df['Date'].dt.to_period('M')
df['Year']  = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

print("\n" + "=" * 50)
print("3. 전처리 후 - 결측치 통계")
print("=" * 50)
print(df.isnull().sum())
print(f"\n전처리 후 데이터: {df.shape[0]:,} 행")


# =====================
# 4. 카테고리 선택 및 월별 집계
# =====================
top_category = df['Product_Category'].value_counts().index[0]
filtered = df[df['Product_Category'] == top_category]

monthly = (
    filtered
    .groupby('YearMonth')['Order_Demand']
    .sum()
    .reset_index()
    .sort_values('YearMonth')
)
monthly['date']  = monthly['YearMonth'].dt.to_timestamp()
monthly['year']  = monthly['YearMonth'].dt.year
monthly['month'] = monthly['YearMonth'].dt.month

print(f"\n선택 카테고리: {top_category}")
print(f"월별 데이터: {len(monthly)}개월")


# =====================
# 5. Histogram - Order_Demand 분포
# =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 전체 주문량 분포
axes[0].hist(df['Order_Demand'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Order_Demand 전체 분포 (Histogram)', fontsize=13)
axes[0].set_xlabel('주문량')
axes[0].set_ylabel('빈도')
axes[0].axvline(df['Order_Demand'].mean(), color='orange', linestyle='--',
                label=f"평균: {df['Order_Demand'].mean():,.0f}")
axes[0].legend()

# 월별 집계 주문량 분포
axes[1].hist(monthly['Order_Demand'], bins=20, color='coral', edgecolor='white')
axes[1].set_title(f'{top_category} 월별 주문량 분포', fontsize=13)
axes[1].set_xlabel('월별 주문량')
axes[1].set_ylabel('빈도')
axes[1].axvline(monthly['Order_Demand'].mean(), color='navy', linestyle='--',
                label=f"평균: {monthly['Order_Demand'].mean():,.0f}")
axes[1].legend()

plt.tight_layout()
plt.savefig('01_histogram.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 01_histogram.png 저장 완료")


# =====================
# 6. Heatmap - 연도 x 월 계절성
# =====================
pivot = monthly.pivot_table(
    values='Order_Demand', index='year', columns='month', aggfunc='sum'
)
pivot.columns = ['1월','2월','3월','4월','5월','6월',
                 '7월','8월','9월','10월','11월','12월']

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
            linewidths=0.5, ax=ax)
ax.set_title(f'{top_category} — 연도×월 주문량 히트맵 (계절성 패턴)', fontsize=13)
ax.set_xlabel('월')
ax.set_ylabel('연도')
plt.tight_layout()
plt.savefig('02_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 02_heatmap.png 저장 완료")


# =====================
# 7. 월별 주문량 추이
# =====================
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(monthly['date'], monthly['Order_Demand'],
        color='steelblue', linewidth=1.8, label='실제 주문량')
ax.fill_between(monthly['date'], monthly['Order_Demand'],
                alpha=0.2, color='steelblue')
avg = monthly['Order_Demand'].mean()
ax.axhline(avg, color='orange', linestyle='--',
           label=f'평균 ({avg:,.0f})')
ax.set_title(f'{top_category} — 월별 주문량 추이', fontsize=13)
ax.set_xlabel('날짜')
ax.set_ylabel('주문량')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('03_trend.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 03_trend.png 저장 완료")


# =====================
# 8. 데이터 전처리 - 시퀀스 생성
# =====================
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TIME_STEPS = 12
values = monthly['Order_Demand'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

X, y = [], []
for i in range(len(scaled) - TIME_STEPS):
    X.append(scaled[i : i + TIME_STEPS])
    y.append(scaled[i + TIME_STEPS])
X, y = np.array(X), np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\nX_train: {X_train.shape}, X_test: {X_test.shape}")


# =====================
# 9. 모델 정의 및 컴파일
# =====================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def build_model(layer_type, units1=64, units2=32):
    model = Sequential()
    if layer_type == 'SimpleRNN':
        model.add(SimpleRNN(units1, return_sequences=True,
                            input_shape=(TIME_STEPS, 1)))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(units2))
    elif layer_type == 'LSTM':
        model.add(LSTM(units1, return_sequences=True,
                       input_shape=(TIME_STEPS, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units2))
    elif layer_type == 'GRU':
        model.add(GRU(units1, return_sequences=True,
                      input_shape=(TIME_STEPS, 1)))
        model.add(Dropout(0.2))
        model.add(GRU(units2))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

es = EarlyStopping(patience=15, restore_best_weights=True)

def evaluate_model(y_true, y_pred):
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1,1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
    return {
        'RMSE': mean_squared_error(y_true_inv, y_pred_inv) ** 0.5,
        'MAE':  mean_absolute_error(y_true_inv, y_pred_inv),
        'R2':   r2_score(y_true_inv, y_pred_inv)
    }, y_true_inv, y_pred_inv


# =====================
# 10. 모델 학습 (SimpleRNN)
# =====================
print("\n[SimpleRNN 학습 중...]")
rnn = build_model('SimpleRNN')
rnn_hist = rnn.fit(X_train, y_train, epochs=150, batch_size=8,
                   validation_split=0.1, verbose=0, callbacks=[es])
rnn_pred = rnn.predict(X_test, verbose=0)
rnn_score, y_true_inv, rnn_inv = evaluate_model(y_test, rnn_pred)
print(f"SimpleRNN - RMSE: {rnn_score['RMSE']:,.2f} | MAE: {rnn_score['MAE']:,.2f} | R2: {rnn_score['R2']:.4f}")


# =====================
# 11. 모델 학습 (LSTM)
# =====================
print("\n[LSTM 학습 중...]")
lstm = build_model('LSTM')
lstm_hist = lstm.fit(X_train, y_train, epochs=150, batch_size=8,
                     validation_split=0.1, verbose=0, callbacks=[es])
lstm_pred = lstm.predict(X_test, verbose=0)
lstm_score, _, lstm_inv = evaluate_model(y_test, lstm_pred)
print(f"LSTM      - RMSE: {lstm_score['RMSE']:,.2f} | MAE: {lstm_score['MAE']:,.2f} | R2: {lstm_score['R2']:.4f}")


# =====================
# 12. 모델 학습 (GRU)
# =====================
print("\n[GRU 학습 중...]")
gru = build_model('GRU')
gru_hist = gru.fit(X_train, y_train, epochs=150, batch_size=8,
                   validation_split=0.1, verbose=0, callbacks=[es])
gru_pred = gru.predict(X_test, verbose=0)
gru_score, _, gru_inv = evaluate_model(y_test, gru_pred)
print(f"GRU       - RMSE: {gru_score['RMSE']:,.2f} | MAE: {gru_score['MAE']:,.2f} | R2: {gru_score['R2']:.4f}")


# =====================
# 13. 학습 곡선 시각화 (Loss)
# =====================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, hist, title in zip(
    axes,
    [rnn_hist, lstm_hist, gru_hist],
    ['SimpleRNN - Loss', 'LSTM - Loss', 'GRU - Loss']
):
    ax.plot(hist.history['loss'], label='Train Loss', color='steelblue')
    ax.plot(hist.history['val_loss'], label='Validation Loss', color='orange')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_loss_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 04_loss_curve.png 저장 완료")


# =====================
# 14. 모델 비교 - Scatter (Actual vs Predicted)
# =====================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, y_pred, title, color in zip(
    axes,
    [rnn_inv, lstm_inv, gru_inv],
    ['SimpleRNN: 실제 vs 예측', 'LSTM: 실제 vs 예측', 'GRU: 실제 vs 예측'],
    ['#4C72B0', '#55A868', '#C44E52']
):
    ax.scatter(y_true_inv, y_pred, alpha=0.4, color=color)
    ax.plot([y_true_inv.min(), y_true_inv.max()],
            [y_true_inv.min(), y_true_inv.max()], 'r--', linewidth=1.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('실제값')
    ax.set_ylabel('예측값')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_model_compare_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 05_model_compare_scatter.png 저장 완료")


# =====================
# 15. 성능 비교 바차트
# =====================
results = pd.DataFrame({
    '모델':  ['SimpleRNN', 'LSTM', 'GRU'],
    'RMSE': [rnn_score['RMSE'], lstm_score['RMSE'], gru_score['RMSE']],
    'MAE':  [rnn_score['MAE'],  lstm_score['MAE'],  gru_score['MAE']],
    'R2':   [rnn_score['R2'],   lstm_score['R2'],   gru_score['R2']]
})

print("\n" + "=" * 50)
print("모델 성능 비교")
print("=" * 50)
print(results.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
colors = ['#4C72B0', '#55A868', '#C44E52']

for ax, metric in zip(axes, ['RMSE', 'MAE', 'R2']):
    bars = ax.bar(results['모델'], results[metric], color=colors)
    ax.set_title(f'모델별 {metric} 비교', fontsize=12)
    ax.set_ylabel(metric)
    for bar, val in zip(bars, results[metric]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height(), f'{val:,.2f}',
                ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('06_model_compare_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 06_model_compare_bar.png 저장 완료")


# =====================
# 16. 모델 예측 - 시계열 그래프
# =====================
fig, ax = plt.subplots(figsize=(14, 5))
x_axis = range(len(y_true_inv))

ax.plot(x_axis, y_true_inv, label='실제 주문량',
        color='steelblue', linewidth=2)
ax.plot(x_axis, lstm_inv, label='LSTM 예측',
        color='crimson', linestyle='--', linewidth=1.8)
ax.fill_between(x_axis,
                lstm_inv * 0.85, lstm_inv * 1.15,
                alpha=0.2, color='crimson', label='예측 범위 (±15%)')
ax.set_title('LSTM - 실제값 vs 예측값 (테스트 구간)', fontsize=13)
ax.set_xlabel('Time Step')
ax.set_ylabel('주문량')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('07_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 07_actual_vs_predicted.png 저장 완료")


