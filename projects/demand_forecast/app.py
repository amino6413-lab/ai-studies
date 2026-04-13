import os
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ─────────────────────────────────────────────────────────────
# 페이지 기본 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="수요 예측 대시보드",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 제품 수요 예측 대시보드")
st.markdown("과거 주문 데이터를 분석하여 **미래 수요를 예측**하고 **적정 재고**를 제안합니다.")

# ─────────────────────────────────────────────────────────────
# 데이터 로드 및 전처리 (캐싱으로 한 번만 실행)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'archive', 'Historical Product Demand.csv')
    df = pd.read_csv(path)

    # Order_Demand 정리
    # "(100)" 형태 → 음수, "100 " 형태 → 양수 숫자로 변환
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

    # 결측치 및 음수 주문 제거
    df = df.dropna(subset=['Date', 'Order_Demand'])
    df = df[df['Order_Demand'] > 0]

    # 날짜 파생 변수 생성
    df['YearMonth'] = df['Date'].dt.to_period('M')
    df['Year']  = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    return df

df = load_data()

# ─────────────────────────────────────────────────────────────
# 사이드바 - 사용자 선택 옵션
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ 설정")

categories  = sorted(df['Product_Category'].unique())
warehouses  = ['전체'] + sorted(df['Warehouse'].unique())

selected_category  = st.sidebar.selectbox("카테고리 선택", categories)
selected_warehouse = st.sidebar.selectbox("창고 선택", warehouses)
predict_months     = st.sidebar.slider("예측 기간 (개월)", min_value=3, max_value=12, value=6)

st.sidebar.markdown("---")
st.sidebar.markdown("**모델:** LSTM (Long Short-Term Memory)")
st.sidebar.markdown("**학습 데이터:** 2011 ~ 2017년")

# ─────────────────────────────────────────────────────────────
# 데이터 필터링
# ─────────────────────────────────────────────────────────────
if selected_warehouse == '전체':
    filtered = df[df['Product_Category'] == selected_category].copy()
else:
    filtered = df[
        (df['Product_Category'] == selected_category) &
        (df['Warehouse'] == selected_warehouse)
    ].copy()

# 월별 합계 집계
monthly = (
    filtered
    .groupby('YearMonth')['Order_Demand']
    .sum()
    .reset_index()
    .sort_values('YearMonth')
)
monthly['date'] = monthly['YearMonth'].dt.to_timestamp()
monthly['year']  = monthly['YearMonth'].dt.year
monthly['month'] = monthly['YearMonth'].dt.month

# ─────────────────────────────────────────────────────────────
# 탭 구성
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 수요 트렌드", "🔮 수요 예측 (LSTM)", "📦 적정 재고 분석"])

# ═══════════════════════════════════════════════════
# TAB 1 : 수요 트렌드
# ═══════════════════════════════════════════════════
with tab1:

    # 요약 지표
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 주문량",     f"{filtered['Order_Demand'].sum():,.0f}")
    col2.metric("월 평균 주문량", f"{monthly['Order_Demand'].mean():,.0f}")
    col3.metric("월 최대 주문량", f"{monthly['Order_Demand'].max():,.0f}")
    col4.metric("월 최소 주문량", f"{monthly['Order_Demand'].min():,.0f}")

    st.markdown("---")

    # 월별 주문량 추이 그래프
    st.subheader("📊 월별 주문량 추이")
    fig1, ax1 = plt.subplots(figsize=(13, 4))
    ax1.plot(monthly['date'], monthly['Order_Demand'],
             color='steelblue', linewidth=1.8, label='실제 주문량')
    ax1.fill_between(monthly['date'], monthly['Order_Demand'], alpha=0.25, color='steelblue')
    avg_line = monthly['Order_Demand'].mean()
    ax1.axhline(avg_line, color='orange', linestyle='--', linewidth=1.2, label=f'평균 ({avg_line:,.0f})')
    ax1.set_xlabel("날짜")
    ax1.set_ylabel("주문량")
    ax1.set_title(f"{selected_category} | {selected_warehouse} — 월별 주문량 추이")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    plt.close()

    # 연도별 x 월별 히트맵 (계절성 확인)
    st.subheader("🗓️ 연도 × 월 주문량 히트맵 (계절성 패턴)")
    pivot = monthly.pivot_table(
        values='Order_Demand', index='year', columns='month', aggfunc='sum'
    )
    pivot.columns = ['1월','2월','3월','4월','5월','6월',
                     '7월','8월','9월','10월','11월','12월']

    fig2, ax2 = plt.subplots(figsize=(13, 4))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
                linewidths=0.5, ax=ax2)
    ax2.set_title("연도별 월별 주문량 (색이 진할수록 수요 높음)")
    ax2.set_xlabel("월")
    ax2.set_ylabel("연도")
    st.pyplot(fig2)
    plt.close()

    st.info("💡 **히트맵 보는 법**: 색이 진한 달 = 수요 많음 → 생산 미리 증량 필요")


# ═══════════════════════════════════════════════════
# TAB 2 : 수요 예측 (LSTM)
# ═══════════════════════════════════════════════════
with tab2:
    st.subheader(f"🔮 향후 {predict_months}개월 수요 예측")

    if len(monthly) < 15:
        st.warning("⚠️ 데이터가 부족합니다 (최소 15개월 필요). 다른 카테고리 또는 '전체' 창고를 선택해 주세요.")
    else:
        st.markdown(f"""
        - **학습 데이터**: {monthly['date'].min().strftime('%Y년 %m월')} ~ {monthly['date'].max().strftime('%Y년 %m월')}
        - **모델**: LSTM (과거 12개월 패턴 → 다음 달 예측)
        - **예측 기간**: {predict_months}개월
        """)

        if st.button("🚀 예측 실행", type="primary"):
            with st.spinner("LSTM 모델 학습 중... (약 10~20초)"):

                TIME_STEPS = 12  # 과거 12개월을 보고 다음 달 예측

                values = monthly['Order_Demand'].values.reshape(-1, 1)

                # 정규화 (0~1 사이로 압축)
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(values)

                # 시퀀스 생성
                X, y = [], []
                for i in range(len(scaled) - TIME_STEPS):
                    X.append(scaled[i : i + TIME_STEPS])
                    y.append(scaled[i + TIME_STEPS])
                X, y = np.array(X), np.array(y)

                # LSTM 모델 구성
                model = Sequential([
                    LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, 1)),
                    Dropout(0.2),
                    LSTM(32, return_sequences=False),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')

                # 학습 (EarlyStopping으로 과적합 방지)
                model.fit(
                    X, y,
                    epochs=150,
                    batch_size=8,
                    verbose=0,
                    callbacks=[EarlyStopping(patience=15, restore_best_weights=True)]
                )

                # 미래 predict_months 개월 예측
                current_seq = scaled[-TIME_STEPS:].copy()
                predictions = []

                for _ in range(predict_months):
                    pred = model.predict(
                        current_seq.reshape(1, TIME_STEPS, 1), verbose=0
                    )
                    predictions.append(pred[0, 0])
                    # 시퀀스를 한 칸 밀고 예측값 추가
                    current_seq = np.append(current_seq[1:], pred).reshape(-1, 1)

                # 역정규화 (0~1 → 원래 주문량 단위로)
                pred_values = scaler.inverse_transform(
                    np.array(predictions).reshape(-1, 1)
                )
                pred_values = np.maximum(pred_values, 0)  # 음수 방지

                # 예측 날짜 생성
                last_date  = monthly['date'].max()
                pred_dates = pd.date_range(last_date, periods=predict_months + 1, freq='MS')[1:]

            st.success("✅ 예측 완료!")

            # 예측 결과 그래프
            fig3, ax3 = plt.subplots(figsize=(13, 5))
            ax3.plot(monthly['date'], monthly['Order_Demand'],
                     color='steelblue', linewidth=1.8, label='실제 수요')
            ax3.plot(pred_dates, pred_values,
                     color='crimson', linewidth=2.2, linestyle='--',
                     marker='o', markersize=6, label='예측 수요')
            # 예측 불확실성 범위 (±15%)
            ax3.fill_between(
                pred_dates,
                pred_values.flatten() * 0.85,
                pred_values.flatten() * 1.15,
                alpha=0.2, color='crimson', label='예측 범위 (±15%)'
            )
            ax3.axvline(x=last_date, color='gray', linestyle=':', alpha=0.6, label='예측 시작점')
            ax3.set_title(f"{selected_category} 수요 예측 결과")
            ax3.set_xlabel("날짜")
            ax3.set_ylabel("주문량")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
            plt.close()

            # 예측 수치 테이블
            st.subheader("📋 월별 예측 결과")
            monthly_avg = monthly['Order_Demand'].mean()

            pred_df = pd.DataFrame({
                '예측 월':   pred_dates.strftime('%Y년 %m월'),
                '예측 수요량': pred_values.flatten().astype(int),
            })

            # 평균 대비 상태 판단
            def status(val):
                if val > monthly_avg * 1.3:
                    return "⚠️ 급등 예상 — 생산 증량 권장"
                elif val < monthly_avg * 0.7:
                    return "📉 감소 예상 — 생산 축소 검토"
                else:
                    return "✅ 정상 범위"

            pred_df['상태'] = pred_df['예측 수요량'].apply(status)
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            # 경고 메시지
            max_pred = pred_values.max()
            if max_pred > monthly_avg * 1.3:
                st.warning(f"⚠️ 예측 기간 중 최대 **{max_pred:,.0f}개** 수요 발생 가능 → 사전 생산 계획 필요!")


# ═══════════════════════════════════════════════════
# TAB 3 : 적정 재고 분석
# ═══════════════════════════════════════════════════
with tab3:
    st.subheader("📦 안전재고 & 적정 버퍼 물량 분석")

    monthly_avg = monthly['Order_Demand'].mean()
    monthly_std = monthly['Order_Demand'].std()
    monthly_max = monthly['Order_Demand'].max()
    monthly_min = monthly['Order_Demand'].min()

    # 서비스 레벨별 안전재고 계산
    # 공식: 안전재고 = Z값 × 수요 표준편차
    safety_90 = 1.282 * monthly_std  # 90% 서비스 레벨
    safety_95 = 1.645 * monthly_std  # 95% 서비스 레벨 (일반 권장)
    safety_99 = 2.326 * monthly_std  # 99% 서비스 레벨 (고위험)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 수요 통계 요약")
        stats_df = pd.DataFrame({
            '항목': ['월 평균 주문량', '월 최대 주문량', '월 최소 주문량',
                    '표준편차', '변동계수(CV)'],
            '수치': [
                f"{monthly_avg:,.0f}",
                f"{monthly_max:,.0f}",
                f"{monthly_min:,.0f}",
                f"{monthly_std:,.0f}",
                f"{(monthly_std/monthly_avg)*100:.1f}%"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        st.caption("변동계수(CV)가 높을수록 수요 변동이 심함 → 안전재고 더 필요")

    with col2:
        st.markdown("### 🛡️ 서비스 레벨별 권장 안전재고")
        safety_df = pd.DataFrame({
            '서비스 레벨': ['90%', '95% ⭐ 권장', '99%'],
            '안전재고량':  [f"{safety_90:,.0f}", f"{safety_95:,.0f}", f"{safety_99:,.0f}"],
            '의미':        [
                '10번 중 1번 재고 부족 감수',
                '20번 중 1번 재고 부족 감수',
                '100번 중 1번 재고 부족 감수'
            ]
        })
        st.dataframe(safety_df, use_container_width=True, hide_index=True)

    # 핵심 권장사항
    surplus_ratio = ((monthly_max / monthly_avg) - 1) * 100
    st.info(f"""
    💡 **{selected_category} 권장 생산 대응 가이드**

    - ✅ 평상시 기준 생산량: **{monthly_avg:,.0f}개/월**
    - 🛡️ 최소 안전재고 유지: **{safety_95:,.0f}개** (95% 서비스 레벨 기준)
    - ⚠️ 최대 수요 대비 여유분: **{(monthly_max - monthly_avg):,.0f}개** (평균 대비 +{surplus_ratio:.0f}%)
    - 📦 총 권장 보유 재고: **{(monthly_avg + safety_95):,.0f}개**
    """)

    st.markdown("---")

    # 월별 수요 분포 박스플롯 (계절성 이상치 확인)
    st.subheader("📅 월별 수요 분포 (박스플롯)")
    month_label = {1:'1월',2:'2월',3:'3월',4:'4월',5:'5월',6:'6월',
                   7:'7월',8:'8월',9:'9월',10:'10월',11:'11월',12:'12월'}
    monthly['month_label'] = monthly['month'].map(month_label)

    fig4, ax4 = plt.subplots(figsize=(13, 4))
    order = list(month_label.values())
    monthly.boxplot(
        column='Order_Demand',
        by='month_label',
        ax=ax4,
        positions=range(1, 13)
    )
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(order, fontsize=9)
    ax4.axhline(monthly_avg, color='orange', linestyle='--',
                linewidth=1.2, label=f'평균 ({monthly_avg:,.0f})')
    ax4.axhline(monthly_avg + safety_95, color='red', linestyle=':',
                linewidth=1.0, label=f'안전재고선 ({monthly_avg + safety_95:,.0f})')
    ax4.set_title("월별 주문량 분포 — 박스가 높은 달 = 수요 불안정")
    ax4.set_xlabel("월")
    ax4.set_ylabel("주문량")
    ax4.legend()
    plt.suptitle("")
    st.pyplot(fig4)
    plt.close()

    st.caption("박스가 넓을수록 그 달의 수요 변동이 심함 → 해당 달 전달부터 재고 선확보 필요")
