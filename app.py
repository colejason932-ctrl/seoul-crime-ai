import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import os
pip install --upgrade google-generativeai

# ------------------------------------------
# [추가됨] 구글 Gemini API 라이브러리 추가
# ------------------------------------------
import google.generativeai as genai

# ==========================================
# 1. 페이지 설정 및 디자인
# ==========================================
st.set_page_config(page_title="서울시 범죄 안전 AI 시스템", layout="wide")

st.markdown("""
    <style>
    .main { background-color: transparent; }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        border-left: 5px solid #0984e3;
    }
    .consulting-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-container { display: flex; justify-content: center; margin-bottom: 60px; margin-top: 30px; }
    .hero-box {
        background: rgba(30, 39, 46, 0.6);
        backdrop-filter: blur(15px);
        padding: 40px 60px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        text-align: center;
        width: fit-content;
    }
    .hero-title { 
        font-size: 34px; font-weight: 900; 
        background: linear-gradient(90deg, #74b9ff, #a29bfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 15px; 
    }
    .hero-sub { font-size: 17px; color: #b2bec3; line-height: 1.8; }
    button[kind="primary"] {
        width: 100% !important; height: 240px !important; border-radius: 20px !important;
        background: rgba(45, 52, 54, 0.8) !important; backdrop-filter: blur(10px) !important;
        color: #dfe6e9 !important; border: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2) !important; font-size: 19px !important;
        font-weight: 700 !important; white-space: pre-wrap !important; line-height: 1.7 !important;
        transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    }
    button[kind="primary"]:hover { 
        transform: translateY(-12px) !important; 
        background: linear-gradient(135deg, #0984e3 0%, #6c5ce7 100%) !important; 
        color: #ffffff !important; border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 15px 35px rgba(108, 92, 231, 0.5) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# [추가됨] 1.5. Gemini AI 챗봇 API 환경 설정
# ==========================================
# 여기에 발급받으신 API 키를 그대로 입력하시면 됩니다.
genai.configure(api_key="AIzaSyDkjj1GMOLDokt8hzPlRvz6pZmPjPWbngk") 
# 빠른 응답 속도를 가진 gemini-1.5-flash 모델 적용
gemini_model = genai.GenerativeModel('gemini-1.5-flash-001') 

# ==========================================
# 2. 데이터 및 AI 모델 로드 (캐싱)
# ==========================================
@st.cache_resource
def load_ai_engine():
    data_path = 'Seoul_Crime_Model_Data.csv'
    model_path = 'xgb_crime_model.json'
    try:
        df = pd.read_csv(data_path)
        df = df[df['발생_10만명당'] > 0].copy()
        features = [
            '야간_유동인구', '버스노선_총합', '노후주택_비율', '고시원_수', 
            'CCTV_대수', '조도_지수', '자연감시_시너지_지수', '사각지대_취약_지수', '군중밀집_위험_지수'
        ]
        scaler = MinMaxScaler()
        scaler.fit(df[features])
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

# 챗봇 세션 상태 초기화
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

def go_menu(menu_name):
    st.session_state.selected_menu = menu_name

# --- 홈 화면 ---
if st.session_state.selected_menu == "home":
    st.markdown("""
        <div class="hero-container">
            <div class="hero-box">
                <div class="hero-title">서울시 범죄 안전 AI 시스템 (XGBoost)</div>
                <div class="hero-sub">
                    CPTED(범죄예방환경설계) 이론과 머신러닝이 결합된 진정한 데이터 주도형 행정 대시보드입니다.<br>
                    하단의 스마트 위젯을 클릭하여 분석을 시작하세요.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🧠\n\nAI 딥다이브\n위험 요인 분석", key="go_deepdive", type="primary"): go_menu("deepdive"); st.rerun()
    with c2:
        if st.button("🎯\n\nCCTV 설치\n우선순위 리포트", key="go_budget", type="primary"): go_menu("budget"); st.rerun()
    with c3:
        if st.button("📊\n\n자치구별\n심층 진단 리포트", key="go_report", type="primary"): go_menu("report"); st.rerun()
    with c4:
        if st.button("🔮\n\n미래 정책\nAI 시뮬레이션", key="go_future", type="primary"): go_menu("future"); st.rerun()
    st.stop()

# --- 공통 상단 ---
top1, top2 = st.columns([8, 1])
with top1: st.title("서울시 범죄 안전 AI 시스템")
with top2:
    if st.button("🏠 홈으로", use_container_width=True):
        st.session_state.selected_menu = "home"; st.rerun()

# 공통 자치구 선택
selected_gu = st.selectbox("분석할 자치구를 선택하세요", sorted(df['자치구'].unique()))
gu_data = df[df['자치구'] == selected_gu].iloc[0]

# ==========================================
# 4. 메뉴 1: AI 딥다이브 
# ==========================================
if st.session_state.selected_menu == "deepdive":
    st.subheader(f"🧠 AI 딥다이브: {selected_gu} 환경 요인 분석")
    st.divider()
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.metric("10만명당 범죄 발생 수", f"{gu_data['발생_10만명당']:.1f}건")
        st.warning("**[AI 모델 분석 원리]**\n본 모델은 단순 상관관계를 넘어, 유동인구와 인프라 간의 상호작용(CPTED)을 비선형적으로 학습한 XGBoost 알고리즘을 사용합니다.")
        st.markdown(f"""
        <div class='consulting-box' style='border-left: 5px solid #0984e3;'>
            <b>📍 {selected_gu} 주요 지표 현황:</b><br><br>
            • 야간 유동인구: <b>{int(gu_data['야간_유동인구']):,}</b>명<br>
            • 노후주택 비율: <b>{gu_data['노후주택_비율']:.1f}</b>%<br>
            • CCTV 대수: <b>{int(gu_data['CCTV_대수'])}</b>대<br>
            • 조도 지수: <b>{gu_data['조도_지수']:.1f}</b>점
        </div>
        """, unsafe_allow_html=True)
    with col2:
        imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="XGBoost 범죄 예측 요인 기여도 (Global)")
        fig_imp.update_traces(marker_color='#6c5ce7')
        fig_imp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp, use_container_width=True)

# ==========================================
# 5. 메뉴 2: CCTV 우선순위
# ==========================================
elif st.session_state.selected_menu == "budget":
    st.subheader("🎯 치안 예산 최적화: CCTV 설치 우선순위 리포트")
    st.markdown("예측 위험도와 인프라 보강 필요도를 기반으로 예산 투입이 가장 시급한 지역을 선별합니다.")
    
    priority_df = df.sort_values('CCTV_설치_우선순위_점수', ascending=False)
    c1, c2 = st.columns(2)
    with c1: st.error(f"### 🚨 긴급 설치 1순위\n{', '.join(priority_df.head(3)['자치구'].tolist())}")
    with c2: st.success(f"### ✅ 유지/관리 구역\n{', '.join(priority_df.tail(3)['자치구'].tolist())}")
        
    st.divider()
    fig_total = px.bar(priority_df, x='자치구', y='CCTV_설치_우선순위_점수', color='CCTV_설치_우선순위_점수', color_continuous_scale='Purples')
    fig_total.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_total, use_container_width=True)

# ==========================================
# 6. 메뉴 3: 심층 리포트 
# ==========================================
elif st.session_state.selected_menu == "report":
    st.subheader(f"📊 {selected_gu} 범죄 및 치안 인프라 심층 리포트")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("야간 유동인구", f"{int(gu_data['야간_유동인구']/10000)}만 명")
    with col2: st.metric("노후주택 비율", f"{gu_data['노후주택_비율']:.1f}%")
    with col3: st.metric("조도 지수", f"{gu_data['조도_지수']:.1f}점")
    with col4: st.metric("고시원 수", f"{int(gu_data['고시원_수'])}개")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        radar_cols = ['노후주택_비율', '야간_유동인구', '사각지대_취약_지수', '군중밀집_위험_지수', '고시원_수']
        radar_vals = [(gu_data[c] / df[c].max()) * 100 if df[c].max() > 0 else 0 for c in radar_cols]
        fig_radar = go.Figure(go.Scatterpolar(r=radar_vals, theta=['노후주택', '야간인구', '사각지대', '군중밀집', '고시원'], fill='toself', line_color='#0984e3'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100]), bgcolor='rgba(0,0,0,0)'), showlegend=False, title=f"{selected_gu} 환경 취약점 지문 (타 지역 대비 %)", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_radar, use_container_width=True)
    with c2:
        st.markdown(f"""
        <div class="consulting-box" style="border-left: 10px solid #fdcb6e;">
            <h4 style="margin-top:0;">[{selected_gu} AI 진단 리포트]</h4>
            <p style="font-size: 15px; line-height: 1.7;">
                <b>🔍 CPTED 분석 결과:</b><br>
                {selected_gu}의 사각지대 취약 지수는 <b>{gu_data['사각지대_취약_지수']:.2f}</b>이며, 자연감시 시너지 지수는 <b>{gu_data['자연감시_시너지_지수']:.1f}</b>입니다. <br><br>
                <b>📍 AI 권장 액션 플랜:</b><br>
                1. 야간 유동인구가 몰리는 22시~04시 시간대 집중 순찰<br>
                2. 노후주택 밀집 구역 중심의 스마트 보안등 및 CCTV 우선 확충
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 7. 메뉴 4: 미래 정책 시뮬레이터
# ==========================================
elif st.session_state.selected_menu == "future":
    st.subheader(f"🔮 {selected_gu} 미래 치안 예측 시뮬레이터 (AI 연동)")
    st.markdown("XGBoost 모델이 사용자의 정책 변화를 실시간으로 받아 CPTED 파생 변수를 재계산하고, 미래의 범죄 발생 건수를 예측합니다.")

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

    if st.sidebar.button("🚀 시뮬레이션 실행", type="primary"):
        with st.spinner("AI가 새로운 정책 조건에서 예측 중입니다..."):
            new_cctv = orig_cctv + cctv_input
            new_illum = min(100.0, orig_illum + illum_input)
            new_pop = orig_pop * (1 - pop_input / 100.0)

            epsilon = 1e-5
            threshold_pop = 500000
            new_synergy = np.log1p(new_pop) * new_illum
            new_blind = new_pop / ((new_cctv + epsilon) * (new_illum + epsilon))
            new_crowd = (new_pop - threshold_pop) ** 2

            orig_row = gu_data[features].to_frame().T
            new_row = orig_row.copy()
            new_row['CCTV_대수'] = new_cctv; new_row['조도_지수'] = new_illum; new_row['야간_유동인구'] = new_pop
            new_row['자연감시_시너지_지수'] = new_synergy; new_row['사각지대_취약_지수'] = new_blind; new_row['군중밀집_위험_지수'] = new_crowd

            pred_before = max(0, model.predict(scaler.transform(orig_row))[0])
            pred_after = max(0, model.predict(scaler.transform(new_row))[0])
            improvement = pred_before - pred_after

            st.divider()
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("정책 전 예측 발생건수", f"{pred_before:.1f}건")
            with c2: st.metric("정책 후 예측 발생건수", f"{pred_after:.1f}건", delta=f"{-improvement:.1f}건", delta_color="inverse")
            with c3: 
                improve_rate = (improvement / pred_before * 100) if pred_before > 0 else 0
                st.metric("범죄 억제율", f"{improve_rate:.1f}%")
    else:
        st.info("👈 좌측 사이드바에서 정책 파라미터를 조절하고 '시뮬레이션 실행' 버튼을 눌러주세요.")

# ==========================================
# [추가됨] 8. 공통 챗봇 모듈 (모든 페이지 하단에 렌더링)
# ==========================================
st.markdown("---")
st.subheader("💬 AI 치안 정책 보좌관 (질의응답)")
st.caption("현재 보고 계신 자치구의 데이터나 치안 관련 궁금한 점을 질문해보세요.")

# 저장된 대화 기록 출력
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자가 채팅창에 입력했을 때
if prompt := st.chat_input(f"{selected_gu} 치안 데이터에 대해 무엇이든 물어보세요..."):
    # 사용자 메시지 화면 출력 및 세션 저장
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 제미나이 AI 응답 처리
    with st.chat_message("assistant"):
        with st.spinner("AI가 분석 중입니다..."):
            try:
                # 자치구 데이터 프롬프트 컨텍스트로 전달
                context = f"현재 분석 중인 지역은 {selected_gu}입니다. 해당 지역의 야간 유동인구는 {gu_data['야간_유동인구']}명, CCTV는 {gu_data['CCTV_대수']}대입니다. 이 정보를 바탕으로 대답해 주세요. 질문: {prompt}"
                response = gemini_model.generate_content(context)
                
                st.markdown(response.text)
                st.session_state.chat_messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"AI 응답 중 오류가 발생했습니다. API 키가 정확한지 확인해 주세요. (에러: {e})")
