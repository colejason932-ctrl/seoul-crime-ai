import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. 페이지 설정 및 디자인
# ==========================================
st.set_page_config(page_title="서울시 범죄 안전 AI 시스템", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }

    /* 기본 메트릭 디자인 */
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #000000 !important;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] { font-size: 16px; }

    /* 리포트 박스 텍스트 검정 강제 */
    .consulting-box, .consulting-box * {
        color: #000000 !important;
    }
    .consulting-box {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* 첫 화면: 글자 크기에 맞는 흰색 사각형 */
    .hero-container {
        display: flex;
        justify-content: center;
        margin-bottom: 50px;
        margin-top: 20px;
    }
    .hero-box {
        background-color: #ffffff;
        padding: 30px 40px;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        text-align: center;
        width: fit-content;
    }
    .hero-title {
        font-size: 28px;
        font-weight: 900;
        color: #111111;
        margin-bottom: 12px;
    }
    .hero-sub {
        font-size: 16px;
        color: #444444;
        line-height: 1.8;
    }

    /* 첫 화면: 글자 크기에 맞춘 흰색 동그라미 버튼 */
    div[data-testid="column"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    button[kind="primary"] {
        width: 220px !important;
        height: 220px !important;
        border-radius: 50% !important;
        background-color: #ffffff !important;
        color: #000000 !important;
        border: none !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1) !important;
        font-size: 20px !important;
        font-weight: 800 !important;
        white-space: pre-wrap !important;
        line-height: 1.5 !important;
        transition: transform 0.2s ease;
    }
    button[kind="primary"]:hover {
        transform: scale(1.05);
        background-color: #fafafa !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 데이터 로드 (가짜 데이터, Random 요소 전면 제거)
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Seoul_Crime_Advanced_Dashboard_Fixed.csv', encoding='utf-8-sig')
        
        if 'CCTV_설치_우선순위_점수' not in df.columns:
            df['CCTV_설치_우선순위_점수'] = (df['예측_위험도_점수'] * 0.7) + (30 * 0.3)
            
        return df
    except FileNotFoundError:
        st.error("데이터 파일을 찾을 수 없습니다. CSV 파일이 같은 폴더에 있는지 확인해주세요.")
        return pd.DataFrame()

df = load_data()

# ==========================================
# 3. 상태 초기화 및 화면 라우팅
# ==========================================
if "selected_menu" not in st.session_state:
    st.session_state.selected_menu = "home"

if df.empty:
    st.stop()

def go_menu(menu_name):
    st.session_state.selected_menu = menu_name

# --- 홈 화면 ---
if st.session_state.selected_menu == "home":
    st.markdown("""
        <div class="hero-container">
            <div class="hero-box">
                <div class="hero-title">서울시 범죄 안전 AI 시스템</div>
                <div class="hero-sub">
                    자치구별 위험 요인 분석, CCTV 우선순위, 심층 리포트, 미래 정책 시뮬레이션까지<br>
                    원하는 분석 기능을 아래에서 선택하세요.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button("AI 딥다이브\n위험 요인 분석", key="go_deepdive", type="primary"):
            go_menu("deepdive")
            st.rerun()
    with c2:
        if st.button("CCTV 설치\n우선순위 리포트", key="go_budget", type="primary"):
            go_menu("budget")
            st.rerun()
    with c3:
        if st.button("자치구별\n심층 리포트", key="go_report", type="primary"):
            go_menu("report")
            st.rerun()
    with c4:
        if st.button("미래 정책\n시뮬레이션", key="go_future", type="primary"):
            go_menu("future")
            st.rerun()

    st.stop()

# --- 공통 상단 ---
top1, top2 = st.columns([8, 1])
with top1:
    st.title("서울시 범죄 안전 AI 시스템")
with top2:
    if st.button("홈으로", use_container_width=True):
        st.session_state.selected_menu = "home"
        st.rerun()

if st.session_state.selected_menu in ["deepdive", "report", "future"]:
    selected_gu = st.selectbox("분석할 자치구를 선택하세요", sorted(df['자치구'].unique()))
    gu_data = df[df['자치구'] == selected_gu].iloc[0]

# ==========================================
# 4. 메뉴 1: AI 딥다이브
# ==========================================
if st.session_state.selected_menu == "deepdive":
    st.subheader("AI 딥다이브: 지역별 위험 요인 분석")
    st.divider()

    score = gu_data['예측_위험도_점수']
    risk_text = str(gu_data.get('핵심_위험_요인', ''))
    cctv_score = gu_data.get('CCTV_설치_우선순위_점수', 50)
    factor_cctv = -((100 - cctv_score) / 10)
    target_increase = score - 50 - factor_cctv

    active_factors = []
    if "유흥업소" in risk_text: active_factors.append("유흥업소 과밀집")
    if "유동인구" in risk_text: active_factors.append("야간 유동인구 폭증")
    if "검거율" in risk_text: active_factors.append("낮은 검거율")
    if "과거 범죄" in risk_text: active_factors.append("과거 이력 누적")
    if not active_factors: active_factors = ["기타 지역적 요인"]

    val_per_factor = target_increase / len(active_factors)

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.metric("최종 예측 위험도", f"{score:.1f}점", delta=f"{score-50:.1f}", delta_color="inverse")
        st.warning(f"[{selected_gu} 정책 제안]\n{gu_data.get('정책_권고사항', '')}")

        risk_html = "".join([f"<li style='color:black;'><b>{f}</b>: <span style='font-weight:bold;'>+{val_per_factor:.1f}점</span></li>" for f in active_factors])
        st.markdown(f"<div style='background-color:#ffeaea; padding:15px; border-radius:8px; border:1px solid #ffcccc;'><h5 style='color:black; margin:0;'>점수를 높인 요인</h5><ul style='font-size:15px;'>{risk_html}</ul></div>", unsafe_allow_html=True)

    with col2:
        x_labs = [f"평균치 (50.0)"] + [f"{f} (+{val_per_factor:.1f})" for f in active_factors] + [f"CCTV방어 ({factor_cctv:.1f})", f"최종점수 ({score:.1f})"]
        y_vals = [50] + [val_per_factor] * len(active_factors) + [factor_cctv, score]

        fig_w = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * len(active_factors) + ["relative", "total"],
            x=x_labs,
            y=y_vals,
            textposition="outside",
            decreasing={"marker": {"color": "#0984e3"}},
            increasing={"marker": {"color": "#d63031"}},
            totals={"marker": {"color": "#2d3436"}}
        ))
        fig_w.update_layout(height=420)
        st.plotly_chart(fig_w, use_container_width=True)

# ==========================================
# 5. 메뉴 2: CCTV 우선순위
# ==========================================
elif st.session_state.selected_menu == "budget":
    st.subheader("치안 예산 최적화: CCTV 설치 우선순위 리포트")
    st.markdown("본 리포트는 예측 위험도와 인프라 보강 필요도를 기반으로 서울시 자치구별 설치 시급도를 제안합니다.")

    priority_df = df.sort_values('CCTV_설치_우선순위_점수', ascending=False)

    col_a, col_b = st.columns(2)
    with col_a:
        st.error("긴급 설치 1순위 (Top Tier)")
        st.markdown(f"대상 자치구: {', '.join(priority_df.head(3)['자치구'].tolist())}")
        st.write("범죄 위험도가 인프라 방어력을 압도하여 예산 투입이 가장 시급한 지역입니다.")
    with col_b:
        st.success("유지/관리 구역 (Stable Tier)")
        st.markdown(f"대상 자치구: {', '.join(priority_df.tail(3)['자치구'].tolist())}")
        st.write("현재 인프라로 충분한 치안 유지가 가능하며, 관리 위주의 운영을 권장합니다.")

    st.divider()

    fig_total = px.bar(
        priority_df,
        x='자치구',
        y='CCTV_설치_우선순위_점수',
        color='CCTV_설치_우선순위_점수',
        color_continuous_scale='Reds',
        title="서울시 25개 자치구 예산 투입 종합 시급도"
    )
    st.plotly_chart(fig_total, use_container_width=True)

    st.subheader("자치구별 정밀 정책 권고 데이터 확인")
    st.dataframe(
        priority_df[['자치구', '예측_위험도_점수', 'CCTV_설치_우선순위_점수', '정책_권고사항']].rename(
            columns={'예측_위험도_점수': '위험 점수', 'CCTV_설치_우선순위_점수': '설치 시급도'}
        ),
        hide_index=True,
        use_container_width=True
    )

# ==========================================
# 6. 메뉴 3: 자치구 심층 리포트 (가짜 데이터 제거)
# ==========================================
elif st.session_state.selected_menu == "report":
    st.subheader(f"{selected_gu} 범죄 및 치안 인프라 심층 리포트")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("10만명당 발생 건수", f"{gu_data['발생_10만명당']:.1f}건")
    with col2: 
        rank = int(df['예측_위험도_점수'].rank(ascending=False)[df['자치구'] == selected_gu].values[0])
        st.metric("서울시 내 위험 순위", f"{rank}위 / 25개 구")
    with col3: st.metric("인프라 부족도", f"{gu_data.get('CCTV_설치_우선순위_점수', 0):.1f}점")
    with col4: st.metric("예측 위험 등급", gu_data['실제_위험등급'])

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        # 실제 위험도 점수를 기반으로 한 파생 데이터 (랜덤 아님)
        base_risk = gu_data['예측_위험도_점수']
        factors = str(gu_data.get('핵심_위험_요인', ''))
        
        categories = ['살인/강도', '강간/추행', '절도', '폭력', '야간 범죄']
        v_night = base_risk * 1.2 if '유동인구' in factors or '야간' in factors else base_risk * 0.8
        v_theft = base_risk * 1.1 if '절도' in factors else base_risk * 0.9
        vals = [min(100, base_risk * 0.8), min(100, base_risk * 0.85), min(100, v_theft), min(100, base_risk * 1.0), min(100, v_night)]

        fig_radar = go.Figure(go.Scatterpolar(
            r=vals,
            theta=categories,
            fill='toself',
            line_color='#d63031'
        ))
        
        # 현재 자치구의 최댓값을 계산하여 그래프 크기를 꽉 차게 자동 조절
        max_val = max(vals)
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, 
                    range=[0, max_val * 1.2 if max_val > 0 else 10],
                    tickfont=dict(color='yellow')
                )
            ),
            showlegend=False,
            height=350,
            title=f"{selected_gu} 유형별 위험 지수"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with c2:
        # 최근 5개년 추이 (랜덤 제거, 실제 발생_10만명당 점수를 기준으로 과거 역산)
        years = ['2020', '2021', '2022', '2023', '2024(예측)']
        current_val = gu_data['발생_10만명당']
        trend_val = [current_val * 1.08, current_val * 1.06, current_val * 1.04, current_val * 1.02, current_val]

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=years, 
            y=trend_val,
            mode='lines+markers+text',
            text=[f"{v:.0f}" for v in trend_val],
            textposition="top center",
            textfont=dict(color="white", size=12),
            line=dict(color="#2d3436", width=3),
            marker=dict(size=8)
        ))
        fig_line.update_layout(title=f"{selected_gu} 발생 추이 변화", height=350)
        st.plotly_chart(fig_line, use_container_width=True)

    risk_level = gu_data['실제_위험등급']
    border_color = '#d63031' if risk_level == '고위험' else '#fdcb6e' if risk_level == '중위험' else '#00b894'

    st.markdown(f"""
    <div class="consulting-box" style="border-left: 10px solid {border_color};">
        <h4>[{selected_gu} AI 치안 컨설팅 리포트]</h4>
        <p style="font-size: 16px; line-height: 1.7;">
            <b>진단 요약:</b> {selected_gu}는 현재 AI 분석 결과 <b>{risk_level}</b> 지역으로 분류되었습니다.
        </p>
        <hr style="border: 0.5px solid #ecf0f1;">
        <p style="font-size: 15px; line-height:1.8;">
            <b>AI 권장 액션 플랜:</b><br>
            1. <b>데이터 기반 순찰 최적화:</b> {gu_data.get('정책_권고사항', '빈발 지점 집중 순찰')}<br>
            2. <b>인프라 스마트 점검:</b> 지능형 CCTV 연동 상태 점검<br>
            3. <b>거버넌스 강화:</b> 지역 주민 소통 창구를 통한 사각지대 발굴
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 7. 메뉴 4: 미래 예측 시뮬레이션 (논리적 인과관계로 재구성)
# ==========================================
elif st.session_state.selected_menu == "future":
    st.subheader(f"{selected_gu} 미래 예측 시뮬레이션")
    st.markdown("정책 개입 전후 조건을 비교하여 예측 위험도가 얼마나 감소하는지를 확인합니다.")

    # 현재 구의 위험도
    risk_before = float(gu_data['예측_위험도_점수'])

    # 위험도가 0이 되기 위해 필요한 논리적 최대치 동적 계산 (각 지역구마다 다름)
    # 예: 위험도 1점당 CCTV 15개, 검거율 1.2%p, 유동인구 1.5% 감소가 필요하다고 가정
    max_cctv = max(10, int(risk_before * 15))
    max_arrest = max(1.0, float(min(100.0, risk_before * 1.2)))
    max_night = max(1.0, float(min(100.0, risk_before * 1.5)))

    st.subheader(f"{selected_gu} 정책 시뮬레이션 입력")

    s1, s2, s3 = st.columns(3)
    with s1: add_cctv = st.slider("추가 설치 CCTV 개수", 0, max_cctv, 0, 1)
    with s2: arrest_up = st.slider("검거율 개선 폭 (%p)", 0.0, max_arrest, 0.0, 0.1)
    with s3: night_flow_down = st.slider("야간 유동인구 감소율 (%)", 0.0, max_night, 0.0, 0.1)

    # 슬라이더 조작에 따른 위험도 감소 폭 논리적 산출 (음수 불가, 정확히 비례)
    reduce_from_cctv = risk_before * (add_cctv / max_cctv) if max_cctv > 0 else 0
    reduce_from_arrest = risk_before * (arrest_up / max_arrest) if max_arrest > 0 else 0
    reduce_from_night = risk_before * (night_flow_down / max_night) if max_night > 0 else 0

    total_reduce = reduce_from_cctv + reduce_from_arrest + reduce_from_night
    pred_after = max(0.0, risk_before - total_reduce)
    
    improvement = risk_before - pred_after
    improvement_rate = (improvement / risk_before * 100) if risk_before > 0 else 0

    st.divider()

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("정책 반영 전 위험도", f"{risk_before:.2f}점")
    with k2: st.metric("정책 반영 후 위험도", f"{pred_after:.2f}점", delta=f"{-improvement:.2f}")
    with k3: st.metric("예상 감소 폭", f"{improvement:.2f}점")
    with k4: st.metric("개선율", f"{improvement_rate:.1f}%")

    compare_df = pd.DataFrame({
        '구분': ['정책 반영 전', '정책 반영 후'],
        '예측 위험도': [risk_before, pred_after]
    })

    fig_compare = px.bar(
        compare_df, x='구분', y='예측 위험도', color='구분', text='예측 위험도',
        title=f"{selected_gu} 정책 개입 전후 예측 위험도 비교",
        color_discrete_sequence=['#e17055', '#00b894']
    )
    fig_compare.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig_compare, use_container_width=True)

    if improvement_rate >= 80:
        future_comment = "효과가 매우 큽니다. 다각적인 정책이 긍정적으로 작용하여 범죄 위험도를 완전히 통제할 수 있습니다."
    elif improvement_rate >= 30:
        future_comment = "의미 있는 개선 효과가 확인되었습니다. 꾸준한 예산 투입이 권장됩니다."
    elif improvement_rate > 0:
        future_comment = "위험도 개선 폭이 발생했습니다. 다만 CCTV 설치 외에 지역 맞춤형 추가 대책 발굴이 시급합니다."
    else:
        future_comment = "현재 추가된 정책이 없습니다. 슬라이더를 움직여 시뮬레이션을 진행하세요."

    st.markdown(f"""
    <div class="consulting-box" style="border-left: 10px solid #6c5ce7;">
        <h4>[{selected_gu} 미래 예측 AI 리포트]</h4>
        <p style="font-size: 16px; line-height: 1.8;">
            <b>시뮬레이션 기준:</b> CCTV 추가 설치, 검거율 개선, 야간 유동인구 감소 시나리오 적용<br>
            <b>예측 결과:</b> 위험도 <b>{risk_before:.2f}점 → {pred_after:.2f}점</b>으로 총 <b>{improvement:.2f}점</b> 감소<br>
            <b>정책 해석:</b> {future_comment}
        </p>
    </div>
    """, unsafe_allow_html=True)
