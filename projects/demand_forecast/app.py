import os
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="수요 예측 대시보드",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 제품 수요 예측 대시보드")
st.markdown("과거 주문 데이터를 분석하여 **미래 수요를 예측**하고 **적정 재고**를 제안합니다.")

# ─────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'archive', 'Historical Product Demand.csv')
    df = pd.read_csv(path)

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

    return df

df = load_data()

# ─────────────────────────────────────────────────────────────
# 사이드바
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

# ─────────────────────────────────────────────────────────────
# 탭 구성
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 수요 트렌드",
    "🔮 수요 예측 (LSTM)",
    "📦 적정 재고 분석",
    "🤖 AI 생산 전략 조언"
])

# ═══════════════════════════════════════════════════
# TAB 1 : 수요 트렌드
# ═══════════════════════════════════════════════════
with tab1:

    if len(monthly) == 0:
        st.warning("⚠️ 선택한 조건에 데이터가 없습니다. 다른 카테고리 또는 '전체' 창고를 선택해 주세요.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("총 주문량",     f"{filtered['Order_Demand'].sum():,.0f}")
        col2.metric("월 평균 주문량", f"{monthly['Order_Demand'].mean():,.0f}")
        col3.metric("월 최대 주문량", f"{monthly['Order_Demand'].max():,.0f}")
        col4.metric("월 최소 주문량", f"{monthly['Order_Demand'].min():,.0f}")

        st.markdown("---")

        # 월별 추이
        st.subheader("📊 월별 주문량 추이")
        fig1, ax1 = plt.subplots(figsize=(13, 4))
        ax1.plot(monthly['date'], monthly['Order_Demand'],
                 color='steelblue', linewidth=1.8, label='실제 주문량')
        ax1.fill_between(monthly['date'], monthly['Order_Demand'],
                         alpha=0.25, color='steelblue')
        avg_line = monthly['Order_Demand'].mean()
        ax1.axhline(avg_line, color='orange', linestyle='--', linewidth=1.2,
                    label=f'평균 ({avg_line:,.0f})')
        ax1.set_xlabel("날짜")
        ax1.set_ylabel("주문량")
        ax1.set_title(f"{selected_category} | {selected_warehouse} — 월별 주문량 추이")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close()

        # 계절성 히트맵
        st.subheader("🗓️ 연도 × 월 주문량 히트맵 (계절성 패턴)")

        pivot = monthly.pivot_table(
            values='Order_Demand', index='year', columns='month', aggfunc='sum'
        )

        if pivot.empty or len(pivot.columns) == 0:
            st.warning("⚠️ 데이터 부족으로 히트맵을 표시할 수 없습니다. '전체' 창고를 선택해 주세요.")
        else:
            month_names = {1:'1월',2:'2월',3:'3월',4:'4월',5:'5월',6:'6월',
                           7:'7월',8:'8월',9:'9월',10:'10월',11:'11월',12:'12월'}
            pivot.columns = [month_names[c] for c in pivot.columns]

            fig2, ax2 = plt.subplots(figsize=(13, 4))
            sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
                        linewidths=0.5, ax=ax2)
            ax2.set_title("연도별 월별 주문량 (색이 진할수록 수요 높음)")
            ax2.set_xlabel("월")
            ax2.set_ylabel("연도")
            st.pyplot(fig2)
            plt.close()

        st.info("💡 색이 진한 달 = 수요 많음 → 생산 미리 증량 필요")


# ═══════════════════════════════════════════════════
# TAB 2 : 수요 예측 (LSTM)
# ═══════════════════════════════════════════════════
with tab2:
    st.subheader(f"🔮 향후 {predict_months}개월 수요 예측")

    if len(monthly) < 15:
        st.warning("⚠️ 데이터 부족 (최소 15개월 필요). 다른 카테고리 또는 '전체' 창고를 선택해 주세요.")
    else:
        st.markdown(f"""
        - **학습 데이터**: {monthly['date'].min().strftime('%Y년 %m월')} ~ {monthly['date'].max().strftime('%Y년 %m월')}
        - **모델**: LSTM (과거 12개월 패턴 → 다음 달 예측)
        - **예측 기간**: {predict_months}개월
        """)

        if st.button("🚀 예측 실행", type="primary"):
            with st.spinner("LSTM 모델 학습 중... (약 10~20초)"):

                from sklearn.preprocessing import MinMaxScaler
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout
                from tensorflow.keras.callbacks import EarlyStopping

                TIME_STEPS = 12
                values = monthly['Order_Demand'].values.reshape(-1, 1)

                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(values)

                X, y = [], []
                for i in range(len(scaled) - TIME_STEPS):
                    X.append(scaled[i : i + TIME_STEPS])
                    y.append(scaled[i + TIME_STEPS])
                X, y = np.array(X), np.array(y)

                model = Sequential([
                    LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, 1)),
                    Dropout(0.2),
                    LSTM(32, return_sequences=False),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(
                    X, y,
                    epochs=150,
                    batch_size=8,
                    verbose=0,
                    callbacks=[EarlyStopping(patience=15, restore_best_weights=True)]
                )

                current_seq = scaled[-TIME_STEPS:].copy()
                predictions = []

                for _ in range(predict_months):
                    pred = model.predict(
                        current_seq.reshape(1, TIME_STEPS, 1), verbose=0
                    )
                    predictions.append(pred[0, 0])
                    current_seq = np.append(current_seq[1:], pred).reshape(-1, 1)

                pred_values = scaler.inverse_transform(
                    np.array(predictions).reshape(-1, 1)
                )
                pred_values = np.maximum(pred_values, 0)

                last_date  = monthly['date'].max()
                pred_dates = pd.date_range(last_date, periods=predict_months + 1, freq='MS')[1:]

                # 탭4에서 사용하기 위해 세션에 저장
                st.session_state['pred_values'] = pred_values
                st.session_state['pred_dates']  = pred_dates
                st.session_state['monthly_avg'] = monthly['Order_Demand'].mean()
                st.session_state['monthly_std'] = monthly['Order_Demand'].std()
                st.session_state['category']    = selected_category

            st.success("✅ 예측 완료!")

            fig3, ax3 = plt.subplots(figsize=(13, 5))
            ax3.plot(monthly['date'], monthly['Order_Demand'],
                     color='steelblue', linewidth=1.8, label='실제 수요')
            ax3.plot(pred_dates, pred_values,
                     color='crimson', linewidth=2.2, linestyle='--',
                     marker='o', markersize=6, label='예측 수요')
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

            st.subheader("📋 월별 예측 결과")
            monthly_avg = monthly['Order_Demand'].mean()

            pred_df = pd.DataFrame({
                '예측 월':    pred_dates.strftime('%Y년 %m월'),
                '예측 수요량': pred_values.flatten().astype(int),
            })

            def status(val):
                if val > monthly_avg * 1.3:
                    return "⚠️ 급등 예상 — 생산 증량 권장"
                elif val < monthly_avg * 0.7:
                    return "📉 감소 예상 — 생산 축소 검토"
                else:
                    return "✅ 정상 범위"

            pred_df['상태'] = pred_df['예측 수요량'].apply(status)
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            max_pred = pred_values.max()
            if max_pred > monthly_avg * 1.3:
                st.warning(f"⚠️ 예측 기간 중 최대 **{max_pred:,.0f}개** 수요 발생 가능 → 사전 생산 계획 필요!")


# ═══════════════════════════════════════════════════
# TAB 3 : 적정 재고 분석
# ═══════════════════════════════════════════════════
with tab3:
    st.subheader("📦 안전재고 & 적정 버퍼 물량 분석")

    if len(monthly) == 0:
        st.warning("⚠️ 데이터가 없습니다. 다른 카테고리 또는 '전체' 창고를 선택해 주세요.")
    else:
        monthly_avg = monthly['Order_Demand'].mean()
        monthly_std = monthly['Order_Demand'].std()
        monthly_max = monthly['Order_Demand'].max()
        monthly_min = monthly['Order_Demand'].min()

        safety_90 = 1.282 * monthly_std
        safety_95 = 1.645 * monthly_std
        safety_99 = 2.326 * monthly_std

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

        surplus_ratio = ((monthly_max / monthly_avg) - 1) * 100
        st.info(f"""
        💡 **{selected_category} 권장 생산 대응 가이드**
        - ✅ 평상시 기준 생산량: **{monthly_avg:,.0f}개/월**
        - 🛡️ 최소 안전재고 유지: **{safety_95:,.0f}개** (95% 서비스 레벨 기준)
        - ⚠️ 최대 수요 대비 여유분: **{(monthly_max - monthly_avg):,.0f}개** (평균 대비 +{surplus_ratio:.0f}%)
        - 📦 총 권장 보유 재고: **{(monthly_avg + safety_95):,.0f}개**
        """)

        st.markdown("---")

        st.subheader("📅 월별 수요 분포 (박스플롯)")

        if len(monthly) >= 12:
            month_label = {1:'1월',2:'2월',3:'3월',4:'4월',5:'5월',6:'6월',
                           7:'7월',8:'8월',9:'9월',10:'10월',11:'11월',12:'12월'}
            monthly['month_label'] = monthly['month'].map(month_label)
            unique_months = sorted(monthly['month'].unique())

            fig4, ax4 = plt.subplots(figsize=(13, 4))
            monthly.boxplot(column='Order_Demand', by='month_label', ax=ax4,
                            positions=range(1, len(unique_months) + 1))
            ax4.set_xticks(range(1, len(unique_months) + 1))
            ax4.set_xticklabels([month_label[m] for m in unique_months], fontsize=9)
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
        else:
            st.warning("⚠️ 데이터 부족으로 박스플롯을 표시할 수 없습니다.")


# ═══════════════════════════════════════════════════
# TAB 4 : AI 생산 전략 조언 (LLM - Ollama)
# ═══════════════════════════════════════════════════
with tab4:
    st.subheader("🤖 AI 생산 전략 조언")
    st.markdown("LSTM 예측 결과를 바탕으로 AI가 생산 전략을 조언해드립니다.")

    @st.cache_resource
    def load_llm():
        return OllamaLLM(model="qwen2.5")

    prompt = ChatPromptTemplate.from_template("""
당신은 제조업 생산관리 전문가입니다.
아래 수요 예측 데이터를 분석하여 구체적인 생산 전략을 한국어로 조언해 주세요.

[수요 예측 정보]
- 제품 카테고리: {category}
- 월 평균 수요: {average}개
- 예측 최대 수요: {max_pred}개
- 예측 최소 수요: {min_pred}개
- 수요 상태: {status}
- 권장 안전재고: {safety_stock}개

[조언 항목]
1. 생산 계획 조정 방향
2. 재고 관리 전략
3. 리스크 대응 방안

간결하고 실무적으로 작성해 주세요.
""")

    if 'pred_values' not in st.session_state:
        st.info("💡 먼저 **'수요 예측 (LSTM)'** 탭에서 예측을 실행해 주세요.")
    else:
        pred_values  = st.session_state['pred_values']
        monthly_avg  = st.session_state['monthly_avg']
        monthly_std  = st.session_state['monthly_std']
        category     = st.session_state['category']

        max_pred     = pred_values.max()
        min_pred     = pred_values.min()
        safety_stock = 1.645 * monthly_std

        if max_pred > monthly_avg * 1.3:
            status = "⚠️ 급등 예상 — 사전 생산 증량 필요"
        elif min_pred < monthly_avg * 0.7:
            status = "📉 감소 예상 — 생산 축소 검토"
        else:
            status = "✅ 정상 범위"

        st.markdown(f"""
        **현재 예측 요약**
        - 카테고리: `{category}`
        - 예측 최대: `{max_pred:,.0f}개`
        - 예측 최소: `{min_pred:,.0f}개`
        - 상태: {status}
        """)

        if st.button("🧠 AI 생산 전략 조언 받기", type="primary"):
            with st.spinner("AI가 분석 중입니다..."):
                try:
                    llm   = load_llm()
                    chain = prompt | llm

                    response = chain.invoke({
                        "category":     category,
                        "average":      f"{monthly_avg:,.0f}",
                        "max_pred":     f"{max_pred:,.0f}",
                        "min_pred":     f"{min_pred:,.0f}",
                        "status":       status,
                        "safety_stock": f"{safety_stock:,.0f}",
                    })

                    st.success("✅ AI 조언 완료!")
                    st.markdown("### 📋 생산 전략 조언")
                    st.markdown(response)

                except Exception as e:
                    st.error(f"❌ Ollama 연결 오류: {e}")
                    st.info("Ollama가 실행 중인지 확인해 주세요. (터미널에서 `ollama run qwen2.5` 실행)")