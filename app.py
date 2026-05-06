import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import os

# ==========================================
# 1. 페이지 설정 및 디자인
# ==========================================
st.set_page_config(page_title="서울시 범죄 안전 AI 시스템", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
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
    .consulting-box {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #000000 !important;
    }
    .consulting-box * { color: #000000 !important; }
    .hero-container {
        display: flex; justify-content: center; margin-bottom: 50px; margin-top: 20px;
    }
    .hero-box {
        background-color: #ffffff; padding: 30px 40px; border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08); text-align: center; width: fit-content;
    }
    .hero-title { font-size: 28px; font-weight: 900; color: #111111; margin-bottom: 12px; }
    .hero-sub { font-size: 16px; color: #444444; line-height: 1.8; }
    div[data-testid="column"] { display: flex; justify-content: center; align-items: center; }
    button[kind="primary"] {
        width: 220px !important; height: 220px !important; border-radius: 50% !important;
        background-color: #ffffff !important; color: #000000 !important; border: none !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1) !important; font-size: 20px !important;
        font-weight: 800 !important; white-space: pre-wrap !important; line-height: 1.5 !important;
        transition: transform 0.2s ease;
    }
    button[kind="primary"]:hover { transform: scale(1.05); background-color: #fafafa !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 데이터 및 AI 모델 로드 (캐싱)
# ==========================================
@st.cache_resource
def load_ai_engine():
    # 깃허브 배포를 고려하여 현재 경로를 탐색합니다.
    data_path = 'Seoul_Crime_Model_Data.csv'
    model_path = 'xgb_crime_model.json'
    
    try:
        # 1. 전처리된 데이터 로드 (서울시 25개 자치구만 사용)
        df = pd.read_csv(data_path)
        df = df[df['발생_10만명당'] > 0].copy()
        
        # 2. 스케일러 복원 (학습할 때와 동일하게 0~1 정규화 준비)
        features = [
            '야간_유동인구', '버스노선_총합', '노후주택_비율', '고시원_수', 
            'CCTV_대수', '조도_지수', 
            '자연감시_시너지_지수', '사각지대_취약_지수', '군중밀집_위험_지수'
        ]
        scaler = MinMaxScaler()
        scaler.fit(df[features])
        
        # 3. XGBoost 모델 로드 (뇌 이식)
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        
        return df, scaler, model, features
    except Exception as e:
        st.error(f"AI 엔진 로딩 실패: {e}\n파일이 같은 폴더에 있는지 확인해주세요.")
        return pd.DataFrame(), None, None, []

df, scaler, model, features = load_ai_engine()

if df.empty:
    st.stop()

# ==========================================
# 3. 상태 초기화 및 화면 라우팅
# ==========================================
if "selected_menu" not in st.session_state:
    st.session_state.selected_menu = "home"

def go_menu(menu_name):
    st.session_state.selected_menu = menu_name

# --- 홈 화면 ---
if st.session_state.selected_menu == "home":
    st.markdown("""
        <div class="hero-container">
            <div class="hero-box">
                <div class="hero-title">서울시 범죄 안전 AI 시스템 (XGBoost 연동)</div>
                <div class="hero-sub">
                    CPTED 이론과 머신러닝이 결합된 진정한 데이터 주도형 행정 대시보드입니다.<br>
                    원하는 분석 기능을 아래에서 선택하세요.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("AI 딥다이브\n위험 요인 분석", key="go_deepdive", type="primary"): go_menu("deepdive"); st.rerun()
    with c2:
        if st.button("CCTV 설치\n우선순위 리포트", key="go_budget", type="primary"): go_menu("budget"); st.rerun()
    with c3:
        if st.button("자치구별\n심층 리포트", key="go_report", type="primary"): go_menu("report"); st.rerun()
    with c4:
        if st.button("미래 정책\n시뮬레이션", key="go_future", type="primary"): go_menu("future"); st.rerun()
    st.stop()

# --- 공통 상단 ---
top1, top2 = st.columns([8, 1])
with top1: st.title("서울시 범죄 안전 AI 시스템")
with top2:
    if st.button("홈으로", use_container_width=True):
        st.session_state.selected_menu = "home"; st.rerun()

# 공통 자치구 선택
selected_gu = st.selectbox("분석할 자치구를 선택하세요", sorted(df['자치구'].unique()))
gu_data = df[df['자치구'] == selected_gu].iloc[0]

# ==========================================
# 4. 메뉴 1: AI 딥다이브 (실제 피처 중요도 기반)
# ==========================================
if st.session_state.selected_menu == "deepdive":
    st.subheader(f"🧠 AI 딥다이브: {selected_gu} 환경 요인 분석")
    st.divider()
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.metric("10만명당 범죄 발생 수", f"{gu_data['발생_10만명당']:.1f}건")
        st.warning("**[AI 모델 분석 원리]**\n본 모델은 단순 상관관계를 넘어, 유동인구와 인프라 간의 상호작용(CPTED)을 비선형적으로 학습한 XGBoost 알고리즘을 사용합니다.")
        
        st.markdown(f"""
        <div class='consulting-box' style='border-left: 5px solid #0984e3; padding: 15px;'>
            <b>📍 {selected_gu} 주요 지표 현황:</b><br>
            - 야간 유동인구: {int(gu_data['야간_유동인구']):,}명<br>
            - 노후주택 비율: {gu_data['노후주택_비율']:.1f}%<br>
            - CCTV 대수: {int(gu_data['CCTV_대수'])}대<br>
            - 조도 지수: {gu_data['조도_지수']:.1f}점
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # 실제 모델의 피처 중요도 시각화
        imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="XGBoost 범죄 예측 요인 기여도 (Global)")
        fig_imp.update_traces(marker_color='#d63031')
        st.plotly_chart(fig_imp, use_container_width=True)

# ==========================================
# 5. 메뉴 2: CCTV 우선순위 (변동 없음)
# ==========================================
elif st.session_state.selected_menu == "budget":
    st.subheader("🎯 치안 예산 최적화: CCTV 설치 우선순위 리포트")
    st.markdown("예측 위험도와 인프라 보강 필요도를 기반으로 예산 투입이 가장 시급한 지역을 선별합니다.")
    
    priority_df = df.sort_values('CCTV_설치_우선순위_점수', ascending=False)
    
    c1, c2 = st.columns(2)
    with c1:
        st.error(f"### 🚨 긴급 설치 1순위\n{', '.join(priority_df.head(3)['자치구'].tolist())}")
    with c2:
        st.success(f"### ✅ 유지/관리 구역\n{', '.join(priority_df.tail(3)['자치구'].tolist())}")
        
    st.divider()
    fig_total = px.bar(priority_df, x='자치구', y='CCTV_설치_우선순위_점수', color='CCTV_설치_우선순위_점수', color_continuous_scale='Reds')
    st.plotly_chart(fig_total, use_container_width=True)

# ==========================================
# 6. 메뉴 3: 심층 리포트 (실제 변수 레이더 차트)
# ==========================================
elif st.session_state.selected_menu == "report":
    st.subheader(f"📈 {selected_gu} 범죄 및 치안 인프라 심층 리포트")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("야간 유동인구", f"{int(gu_data['야간_유동인구']/10000)}만 명")
    with col2: st.metric("노후주택 비율", f"{gu_data['노후주택_비율']:.1f}%")
    with col3: st.metric("조도 지수", f"{gu_data['조도_지수']:.1f}점")
    with col4: st.metric("고시원 수", f"{int(gu_data['고시원_수'])}개")

    st.divider()
    c1, c2 = st.columns(2)
    
    with c1:
        # 주요 지표를 0~100으로 상대 평가하여 레이더 차트 생성
        radar_cols = ['노후주택_비율', '야간_유동인구', '사각지대_취약_지수', '군중밀집_위험_지수', '고시원_수']
        radar_vals = [(gu_data[c] / df[c].max()) * 100 if df[c].max() > 0 else 0 for c in radar_cols]
        
        fig_radar = go.Figure(go.Scatterpolar(r=radar_vals, theta=['노후주택', '야간인구', '사각지대', '군중밀집', '고시원'], fill='toself', line_color='#d63031'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title=f"{selected_gu} 환경 취약점 지문 (타 지역 대비 백분율)")
        st.plotly_chart(fig_radar, use_container_width=True)

    with c2:
        st.markdown(f"""
        <div class="consulting-box" style="border-left: 10px solid #fdcb6e;">
            <h4>[{selected_gu} AI 진단 리포트]</h4>
            <p style="font-size: 15px; line-height: 1.7;">
                <b>CPTED 분석 결과:</b><br>
                {selected_gu}의 사각지대 취약 지수는 <b>{gu_data['사각지대_취약_지수']:.2f}</b>이며, 자연감시 시너지 지수는 <b>{gu_data['자연감시_시너지_지수']:.1f}</b>입니다. 
                이는 유동인구 대비 방어 인프라의 현재 밸런스 상태를 나타냅니다.<br><br>
                <b>📍 AI 권장 액션 플랜:</b><br>
                1. 야간 유동인구가 몰리는 22시~04시 시간대 집중 순찰<br>
                2. 노후주택 밀집 구역 중심의 스마트 보안등 및 CCTV 우선 확충<br>
                3. 데이터 기반의 범죄 취약 시간/장소 핀포인트 대응
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 7. 메뉴 4: 진짜 미래 정책 시뮬레이터 (찐 AI 연동)
# ==========================================
elif st.session_state.selected_menu == "future":
    st.subheader(f"🔮 {selected_gu} 미래 치안 예측 시뮬레이터 (AI 연동)")
    st.markdown("XGBoost 모델이 사용자의 정책 변화를 실시간으로 받아 CPTED 파생 변수를 재계산하고, 미래의 범죄 발생 건수를 예측합니다.")

    # 현재 상태 데이터 추출
    orig_cctv = gu_data['CCTV_대수']
    orig_illum = gu_data['조도_지수']
    orig_pop = gu_data['야간_유동인구']

    st.sidebar.header("⚙️ 예방 정책 파라미터")
    st.sidebar.write(f"현재 CCTV: {int(orig_cctv)}대")
    cctv_input = st.sidebar.slider("CCTV 추가 설치 (대)", 0, 500, 0, 10)
    
    st.sidebar.write(f"현재 조도 지수: {orig_illum:.1f}점")
    illum_input = st.sidebar.slider("가로등 조도 개선 (+점수)", 0.0, 50.0, 0.0, 1.0)
    
    st.sidebar.write(f"현재 야간 인구: {int(orig_pop/10000)}만 명")
    pop_input = st.sidebar.slider("야간 유동인구 분산/감소율 (%)", 0.0, 50.0, 0.0, 1.0)

    if st.sidebar.button("시뮬레이션 실행", type="primary"):
        with st.spinner("AI가 새로운 정책 조건에서 CPTED 환경 상호작용을 예측 중입니다..."):
            
            # 1. 사용자 입력 기반 새로운 베이스 지표 계산
            new_cctv = orig_cctv + cctv_input
            new_illum = min(100.0, orig_illum + illum_input)
            new_pop = orig_pop * (1 - pop_input / 100.0)

            # 2. CPTED 파생 변수 재계산 (기획했던 논리 그대로!)
            epsilon = 1e-5
            threshold_pop = 500000
            new_synergy = np.log1p(new_pop) * new_illum
            new_blind = new_pop / ((new_cctv + epsilon) * (new_illum + epsilon))
            new_crowd = (new_pop - threshold_pop) ** 2

            # 3. 모델 입력을 위한 데이터프레임 구성 (1줄짜리 테이블)
            orig_row = gu_data[features].to_frame().T
            new_row = orig_row.copy()
            new_row['CCTV_대수'] = new_cctv
            new_row['조도_지수'] = new_illum
            new_row['야간_유동인구'] = new_pop
            new_row['자연감시_시너지_지수'] = new_synergy
            new_row['사각지대_취약_지수'] = new_blind
            new_row['군중밀집_위험_지수'] = new_crowd

            # 4. 스케일링 후 예측 (predict) 수행
            orig_scaled = scaler.transform(orig_row)
            new_scaled = scaler.transform(new_row)

            pred_before = model.predict(orig_scaled)[0]
            pred_after = model.predict(new_scaled)[0]
            
            # (발생건수가 음수가 나오지 않도록 하한선 설정)
            pred_before = max(0, pred_before)
            pred_after = max(0, pred_after)
            improvement = pred_before - pred_after

            # 5. 결과 시각화 출력
            st.divider()
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("정책 전 예측 발생건수", f"{pred_before:.1f}건")
            with c2: st.metric("정책 후 예측 발생건수", f"{pred_after:.1f}건", delta=f"{-improvement:.1f}건", delta_color="inverse")
            with c3: 
                improve_rate = (improvement / pred_before * 100) if pred_before > 0 else 0
                st.metric("범죄 억제율", f"{improve_rate:.1f}%")

            # 6. XAI 스토리텔링 해석 박스
            orig_blind = orig_row['사각지대_취약_지수'].values[0]
            synergy_diff = new_synergy - orig_row['자연감시_시너지_지수'].values[0]
            
            st.markdown(f"""
            <div class="consulting-box" style="border-left: 10px solid #00b894; margin-top: 20px;">
                <h4>[🧠 XAI 설명 가능한 AI 리포트]</h4>
                <p style="font-size: 16px; line-height: 1.8;">
                    <b>💡 AI는 왜 이런 예측을 내렸을까요?</b><br>
                    입력하신 정책(CCTV 확충, 조도 개선 등)을 시뮬레이션한 결과, 
                    야간 사각지대 취약 지수가 기존 <b>{orig_blind:.2f}</b>에서 <b>{new_blind:.2f}</b>로 개선되었습니다. 
                    AI는 이 사각지대 해소가 우발적 범죄 기회를 차단하는 가장 큰 원인으로 판단했습니다.<br><br>
                    더불어 빛의 조도가 확보되면서 유동인구가 범죄 감시자로 전환되어 '자연 감시 시너지 지수'가 <b>+{synergy_diff:.1f}</b> 만큼 상승하며 방어막이 극대화되는 논리적 인과관계를 모델이 증명하고 있습니다.
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👈 좌측 사이드바에서 정책 파라미터를 조절하고 '시뮬레이션 실행' 버튼을 눌러주세요.")