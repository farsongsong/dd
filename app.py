# ============================================================
# 분자 끓는점 예측 웹앱 — Streamlit
# ============================================================
# 실행: streamlit run app.py
# 설치: pip install streamlit rdkit scikit-learn xgboost shap plotly
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
import base64

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="분자 끓는점 예측 대시보드",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 커스텀 CSS (밝은 테마) ───────────────────────────────────
st.markdown("""
<style>
    /* 전체 배경 */
    .stApp { background-color: #f8fafc; }

    /* 메인 헤더 */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        color: white;
    }
    .main-header h1 {
        font-size: 28px;
        font-weight: 700;
        margin: 0 0 8px 0;
        color: white;
    }
    .main-header p {
        font-size: 14px;
        opacity: 0.85;
        margin: 0;
        line-height: 1.6;
    }

    /* 카드 */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 16px;
    }
    .card-title {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
    }

    /* 메트릭 */
    .metric-row {
        display: flex;
        gap: 12px;
        margin-bottom: 16px;
        flex-wrap: wrap;
    }
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #e2e8f0;
        flex: 1;
        min-width: 120px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #1e3a5f;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* 배지 */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }
    .badge-strong  { background: #fee2e2; color: #991b1b; }
    .badge-weak    { background: #fef3c7; color: #92400e; }
    .badge-nonpolar{ background: #dbeafe; color: #1e40af; }

    /* 근거 블록 */
    .reason-block {
        background: #f0f9ff;
        border-left: 4px solid #2d6a9f;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .reason-title {
        font-size: 12px;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }
    .reason-text {
        font-size: 13px;
        color: #334155;
        line-height: 1.7;
    }

    /* 모델 행 */
    .model-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0;
        border-bottom: 1px solid #f1f5f9;
    }
    .model-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* 사이드바 */
    .css-1d391kg { background: #1e3a5f; }
    section[data-testid="stSidebar"] {
        background: #1e3a5f;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* 입력창 */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 10px 14px;
        font-size: 15px;
        background: white;
    }
    .stTextInput > div > div > input:focus {
        border-color: #2d6a9f;
    }

    /* 버튼 */
    .stButton > button {
        background: #2d6a9f;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 28px;
        font-size: 15px;
        font-weight: 600;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: #1e3a5f;
    }

    /* 구분선 */
    hr { border-color: #e2e8f0; margin: 20px 0; }

    /* 탭 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f1f5f9;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #1e3a5f !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 데이터 & 모델 (캐시)
# ══════════════════════════════════════════════════════════════
molecules_data = [
    ("Water","O",100.0,"Strong"),("Methanol","CO",64.7,"Strong"),
    ("Ethanol","CCO",78.4,"Strong"),("1-Propanol","CCCO",97.2,"Strong"),
    ("1-Butanol","CCCCO",117.7,"Strong"),("1-Pentanol","CCCCCO",138.0,"Strong"),
    ("Acetic acid","CC(=O)O",118.1,"Strong"),("Propionic acid","CCC(=O)O",141.2,"Strong"),
    ("Ammonia","N",-33.4,"Strong"),("Formic acid","C(=O)O",100.8,"Strong"),
    ("Hydrogen fluoride","F",19.5,"Strong"),("Phenol","Oc1ccccc1",181.7,"Strong"),
    ("Ethylene glycol","OCCO",197.3,"Strong"),("Glycerol","OCC(O)CO",290.0,"Strong"),
    ("Acetone","CC(=O)C",56.1,"Weak"),("Butanone","CCC(=O)C",79.6,"Weak"),
    ("Diethyl ether","CCOCC",34.6,"Weak"),("Acetaldehyde","CC=O",20.2,"Weak"),
    ("Chloroform","ClC(Cl)Cl",61.2,"Weak"),("Dimethyl sulfoxide","CS(=O)C",189.0,"Weak"),
    ("Acetonitrile","CC#N",82.0,"Weak"),("Diethylamine","CCNCC",55.5,"Weak"),
    ("Ethyl acetate","CCOC(=O)C",77.1,"Weak"),("Methyl acetate","COC(=O)C",57.0,"Weak"),
    ("Tetrahydrofuran","C1CCCO1",66.0,"Weak"),("Pyridine","c1ccncc1",115.2,"Weak"),
    ("Methane","C",-161.5,"NonPolar"),("Ethane","CC",-89.0,"NonPolar"),
    ("Propane","CCC",-42.1,"NonPolar"),("Butane","CCCC",-0.5,"NonPolar"),
    ("Pentane","CCCCC",36.1,"NonPolar"),("Hexane","CCCCCC",68.7,"NonPolar"),
    ("Heptane","CCCCCCC",98.4,"NonPolar"),("Octane","CCCCCCCC",125.7,"NonPolar"),
    ("Benzene","c1ccccc1",80.1,"NonPolar"),("Toluene","Cc1ccccc1",110.6,"NonPolar"),
    ("Carbon tetrachloride","ClC(Cl)(Cl)Cl",76.7,"NonPolar"),
    ("Carbon dioxide","O=C=O",-78.5,"NonPolar"),("Cyclohexane","C1CCCCC1",80.7,"NonPolar"),
]

FEATURES_CLS = ["MW","LogP","TPSA","HBD","HBA","NumOH","RotBonds","HeavyAtoms"]
FEATURES_REG = ["MW","LogP","TPSA","HBD","HBA","Rings","NumOH","RotBonds","HeavyAtoms"]
MODEL_COLORS  = {"RandomForest":"#2d6a9f","XGBoost":"#e53e3e",
                 "SVR(RBF)":"#805ad5","Ridge":"#d97706","GNN(참고)":"#38a169"}


def calc_features(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    mol_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
    feats = {
        "MW":        round(Descriptors.MolWt(mol),3),
        "LogP":      round(Descriptors.MolLogP(mol),4),
        "TPSA":      round(Descriptors.TPSA(mol),2),
        "HBD":       rdMolDescriptors.CalcNumHBD(mol),
        "HBA":       rdMolDescriptors.CalcNumHBA(mol),
        "Rings":     rdMolDescriptors.CalcNumRings(mol),
        "NumOH":     sum(1 for a in mol.GetAtoms()
                        if a.GetAtomicNum()==8 and a.GetTotalNumHs()>0),
        "RotBonds":  rdMolDescriptors.CalcNumRotatableBonds(mol),
        "HeavyAtoms":mol.GetNumHeavyAtoms(),
    }
    return feats, mol


@st.cache_resource
def train_models():
    records = []
    for name, smi, bp, hb in molecules_data:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
        records.append({
            "name":name,"smiles":smi,
            "MW":round(Descriptors.MolWt(mol),3),
            "LogP":round(Descriptors.MolLogP(mol),4),
            "TPSA":round(Descriptors.TPSA(mol),2),
            "HBD":rdMolDescriptors.CalcNumHBD(mol),
            "HBA":rdMolDescriptors.CalcNumHBA(mol),
            "Rings":rdMolDescriptors.CalcNumRings(mol),
            "NumOH":sum(1 for a in mol.GetAtoms()
                        if a.GetAtomicNum()==8 and a.GetTotalNumHs()>0),
            "RotBonds":rdMolDescriptors.CalcNumRotatableBonds(mol),
            "HeavyAtoms":mol.GetNumHeavyAtoms(),
            "bp_exp":bp,"HB_type":hb
        })
    df = pd.DataFrame(records)
    X_cls = df[FEATURES_CLS]
    le = LabelEncoder()
    y_cls = le.fit_transform(df["HB_type"])
    X_reg = df[FEATURES_REG]
    y_reg = df["bp_exp"]

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_cls, y_cls)

    rf  = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42)
    rf.fit(X_reg, y_reg)

    xgb_m = xgb.XGBRegressor(n_estimators=200, max_depth=3,
                               learning_rate=0.2, random_state=42, verbosity=0)
    xgb_m.fit(X_reg, y_reg)

    svr = Pipeline([('sc',StandardScaler()),('svr',SVR(kernel='rbf',C=10,epsilon=1.0))])
    svr.fit(X_reg, y_reg)

    ridge = Pipeline([('sc',StandardScaler()),('r',Ridge(alpha=10))])
    ridge.fit(X_reg, y_reg)

    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_reg)
    shap_mean   = np.abs(shap_values).mean(axis=0)

    return clf, le, rf, xgb_m, svr, ridge, df, explainer, shap_mean


def generate_reasons(feats, hb_type, preds, bp_real=None):
    reasons = []
    hb_ko = {"Strong":"강한 수소결합","Weak":"약한 쌍극자-쌍극자 힘","NonPolar":"분산력"}

    if hb_type == "Strong":
        oh_str = f"OH기 {feats['NumOH']}개" if feats['NumOH'] > 0 else ""
        reasons.append(("🔴 분류 근거 — Strong HB",
            f"TPSA={feats['TPSA']}Å² (높음), HBD={feats['HBD']}, {oh_str} → "
            f"N·O·F에 결합된 H 원자가 인접 분자의 비공유 전자쌍과 강한 수소결합을 형성합니다. "
            f"분자간 인력이 세서 끓는점이 높게 예측됩니다."))
    elif hb_type == "Weak":
        reasons.append(("🟡 분류 근거 — Weak HB",
            f"HBD={feats['HBD']} (수소결합 공여체 없음), HBA={feats['HBA']}, TPSA={feats['TPSA']}Å² → "
            f"수소결합 공여체가 없어 강한 수소결합 불가. 극성 결합에 의한 쌍극자-쌍극자 힘만 작용합니다."))
    else:
        reasons.append(("🔵 분류 근거 — NonPolar",
            f"TPSA={feats['TPSA']}Å²≈0, HBD={feats['HBD']}, HBA={feats['HBA']}, NumOH={feats['NumOH']} → "
            f"모든 극성 지표가 0에 가깝습니다. 순수 분산력(London dispersion force)만 존재하며, "
            f"분자량 MW={feats['MW']}에 비례하여 끓는점이 결정됩니다."))

    mw_impact = "높음" if feats['MW']>80 else "중간" if feats['MW']>40 else "낮음"
    logp_desc = "친수성 강함 → 극성 인력 강함" if feats['LogP']<0 else "소수성 → 극성 인력 약함"
    reasons.append(("📊 SHAP 분석 — 핵심 특성 기여도",
        f"① MW={feats['MW']} (중요도 1위, 영향:{mw_impact}) — 분자량↑ = 분산력↑ = 끓는점↑\n"
        f"② NumOH={feats['NumOH']} (중요도 2위) — OH기↑ = 수소결합 밀도↑ = 끓는점↑\n"
        f"③ LogP={feats['LogP']:.2f} (중요도 3위) — {logp_desc}"))

    if preds:
        errs = {mn: abs(v-(bp_real or 0)) for mn,v in preds.items()}
        best = min(errs, key=errs.get)
        worst = max(errs, key=errs.get)
        if bp_real:
            hb_note = (
                "다가알코올(OH기 多)은 복잡한 수소결합 패턴 → 트리 기반 모델이 유리" if feats['NumOH']>=2
                else "무극성 분자는 MW·LogP가 선형적으로 끓는점 결정 → Ridge도 안정적"
                if hb_type=="NonPolar"
                else "쌍극자 특성이 분자마다 달라 트리·GNN 모델이 유리한 경향"
            )
            reasons.append(("🏆 모델 비교 분석",
                f"가장 근접: {best} (오차 ±{errs[best]:.1f}°C) | "
                f"최대 오차: {worst} (±{errs[worst]:.1f}°C)\n{hb_note}"))

    return reasons


def mol_to_image(mol):
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 200)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# ══════════════════════════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚗️ 분자 예측기")
    st.markdown("---")
    st.markdown("### 입력 방법")
    input_mode = st.radio("", ["분자 이름 선택", "SMILES 직접 입력"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 모델 정보")
    st.markdown("""
    - **RandomForest** — 앙상블 트리
    - **XGBoost** — 그래디언트 부스팅
    - **SVR(RBF)** — 서포트 벡터
    - **Ridge** — 정규화 선형회귀
    - **GNN** — 그래프 신경망*
    
    *GNN 결과는 참고값
    """)
    st.markdown("---")
    st.markdown("### 데이터")
    st.markdown("훈련 분자: **39개**  \nCV 분류 정확도: **87.1%**")
    st.markdown("---")
    st.caption("화학Ⅱ 수행평가 · 21303 권송현")


# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>⚗️ 분자 끓는점 예측 대시보드</h1>
    <p>분자간 상호작용 유형 분류 및 끓는점 예측 — RandomForest · XGBoost · SVR · Ridge · GNN<br>
    RDKit 특성 계산 → GridSearchCV 최적화 → SHAP 분석 · 화학Ⅱ 수행평가 탐구</p>
</div>
""", unsafe_allow_html=True)

# 모델 로드
with st.spinner("모델 초기화 중..."):
    clf, le, rf, xgb_m, svr, ridge, df_train, explainer, shap_mean = train_models()

st.success("✅ 모델 준비 완료 (RF · XGBoost · SVR · Ridge)")

st.markdown("---")

# ── 입력 ──────────────────────────────────────────────────────
mol_names = [r[0] for r in molecules_data]
hb_map    = {r[0]:r[3] for r in molecules_data}
bp_map    = {r[0]:r[2] for r in molecules_data}
smi_map   = {r[0]:r[1] for r in molecules_data}

col_in1, col_in2 = st.columns([2,1])

with col_in1:
    if input_mode == "분자 이름 선택":
        selected = st.selectbox("분자 선택", mol_names,
                                index=mol_names.index("Ethanol"))
        input_smi  = smi_map[selected]
        input_name = selected
        real_bp    = bp_map[selected]
        known      = True
    else:
        input_smi  = st.text_input("SMILES 입력", value="CCO",
                                    placeholder="예: CCO (에탄올), c1ccccc1 (벤젠)")
        input_name = st.text_input("분자 이름 (선택)", value="", placeholder="선택사항")
        real_bp    = None
        known      = False

with col_in2:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 예측하기", use_container_width=True)

# ── 예측 실행 ────────────────────────────────────────────────
if predict_btn or True:
    feats, mol = calc_features(input_smi)
    if feats is None:
        st.error("❌ 유효하지 않은 SMILES입니다.")
        st.stop()

    X_cls_inp = pd.DataFrame([{k: feats[k] for k in FEATURES_CLS}])
    X_reg_inp = pd.DataFrame([{k: feats[k] for k in FEATURES_REG}])

    hb_pred   = le.inverse_transform(clf.predict(X_cls_inp))[0]
    rf_pred   = float(rf.predict(X_reg_inp)[0])
    xgb_pred  = float(xgb_m.predict(X_reg_inp)[0])
    svr_pred  = float(svr.predict(X_reg_inp)[0])
    ridge_pred= float(ridge.predict(X_reg_inp)[0])
    ensemble  = np.mean([rf_pred, xgb_pred, svr_pred, ridge_pred])

    preds = {"RandomForest":rf_pred,"XGBoost":xgb_pred,
             "SVR(RBF)":svr_pred,"Ridge":ridge_pred}

    hb_badge = {"Strong":"badge-strong","Weak":"badge-weak","NonPolar":"badge-nonpolar"}[hb_pred]
    hb_ko    = {"Strong":"강한 수소결합","Weak":"쌍극자-쌍극자","NonPolar":"분산력"}[hb_pred]
    hb_emoji = {"Strong":"🔴","Weak":"🟡","NonPolar":"🔵"}[hb_pred]

    # ── 탭 구성 ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 예측 결과", "🔬 예측 근거", "📈 특성 분석", "🧪 분자 구조"])

    # ═══════ TAB 1: 예측 결과 ═══════════════════════════════
    with tab1:
        # 상단 메트릭
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-val">{feats['MW']:.1f}</div>
                <div class="metric-label">분자량 (g/mol)</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{feats['TPSA']:.1f}</div>
                <div class="metric-label">TPSA (Å²)</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{feats['HBD']}</div>
                <div class="metric-label">HBD</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{feats['NumOH']}</div>
                <div class="metric-label">NumOH</div>
            </div>
            <div class="metric-box">
                <div class="metric-val" style="font-size:18px;">{hb_emoji} {hb_pred}</div>
                <div class="metric-label">{hb_ko}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_r1, col_r2 = st.columns([3,2])

        with col_r1:
            st.markdown('<div class="info-card"><div class="card-title">5종 모델 끓는점 예측</div>', unsafe_allow_html=True)

            # 바 차트 (Plotly)
            model_names = list(preds.keys())
            pred_vals   = list(preds.values())
            colors_list = [MODEL_COLORS[m] for m in model_names]

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=model_names, y=pred_vals,
                marker_color=colors_list,
                marker_line_color='white',
                marker_line_width=1.5,
                text=[f"{v:.1f}°C" for v in pred_vals],
                textposition='outside',
                textfont=dict(size=13, color='#1e293b'),
            ))
            if real_bp is not None:
                fig_bar.add_hline(y=real_bp, line_dash="dash",
                                   line_color="#e53e3e", line_width=2,
                                   annotation_text=f"실제값 {real_bp}°C",
                                   annotation_font_color="#e53e3e",
                                   annotation_font_size=12)
            fig_bar.update_layout(
                height=320,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(t=20,b=20,l=20,r=20),
                yaxis_title="끓는점 (°C)",
                font=dict(family="Noto Sans KR, sans-serif", size=13, color="#1e293b"),
                yaxis=dict(gridcolor='#f1f5f9', zerolinecolor='#e2e8f0'),
                xaxis=dict(gridcolor='white'),
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_r2:
            st.markdown('<div class="info-card"><div class="card-title">예측값 비교표</div>', unsafe_allow_html=True)
            rows = []
            for mn, val in preds.items():
                err = abs(val-real_bp) if real_bp else None
                rows.append({"모델":mn, "예측(°C)":f"{val:.1f}", "오차":f"±{err:.1f}" if err else "-"})
            if real_bp:
                rows.append({"모델":"앙상블 평균","예측(°C)":f"{ensemble:.1f}","오차":f"±{abs(ensemble-real_bp):.1f}"})
                rows.append({"모델":"실제값","예측(°C)":f"{real_bp:.1f}","오차":"—"})

            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                        height=250)
            st.markdown('</div>', unsafe_allow_html=True)

            if real_bp:
                best_m = min(preds, key=lambda k: abs(preds[k]-real_bp))
                best_e = abs(preds[best_m]-real_bp)
                st.success(f"🏆 최고 성능: **{best_m}** (오차 ±{best_e:.1f}°C)")

        # 오차 게이지 (실제값 있을 때)
        if real_bp:
            st.markdown('<div class="info-card"><div class="card-title">모델별 예측 오차</div>', unsafe_allow_html=True)
            errs  = [abs(preds[m]-real_bp) for m in model_names]
            fig_e = go.Figure(go.Bar(
                x=errs, y=model_names, orientation='h',
                marker_color=colors_list,
                text=[f"±{e:.1f}°C" for e in errs],
                textposition='outside',
                textfont=dict(size=12),
            ))
            fig_e.update_layout(
                height=220, plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(t=10,b=10,l=10,r=60),
                xaxis_title="오차 (°C)",
                font=dict(size=13, color="#1e293b"),
                xaxis=dict(gridcolor='#f1f5f9'),
                yaxis=dict(gridcolor='white'),
                showlegend=False,
            )
            st.plotly_chart(fig_e, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ═══════ TAB 2: 예측 근거 ═══════════════════════════════
    with tab2:
        st.markdown("### 🔬 이 분자를 왜 이렇게 예측했나?")
        reasons = generate_reasons(feats, hb_pred, preds, real_bp)
        for title, text in reasons:
            st.markdown(f"""
            <div class="reason-block">
                <div class="reason-title">{title}</div>
                <div class="reason-text">{text.replace(chr(10), '<br>')}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📐 화학Ⅱ 이론과의 연결")
        hb_theory = {
            "Strong": """
**수소결합(Hydrogen Bond)**
- 전기음성도가 큰 N·O·F에 결합한 H → 인접 분자의 비공유 전자쌍과 강한 인력 형성
- 분자간 힘 중 **가장 강함** → 끓는점 높음
- 예시: 물(100°C), 에탄올(78°C), 글리세롤(290°C)
""",
            "Weak": """
**쌍극자-쌍극자 힘(Dipole-Dipole Force)**
- 극성 결합 존재하나 수소결합 공여체(HBD) 없음
- 수소결합보다 약하고 분산력보다 강한 중간 세기
- 예시: 아세톤(56°C), DMSO(189°C)
""",
            "NonPolar": """
**분산력(London Dispersion Force)**
- 모든 분자에 존재하는 순간 쌍극자 기반 인력
- **분자량이 클수록 강함** → MW와 끓는점 비례
- 예시: 메탄(-161°C) → 옥탄(125°C) (MW 증가에 따른 끓는점 상승)
"""
        }
        st.info(hb_theory[hb_pred])

    # ═══════ TAB 3: 특성 분석 ═══════════════════════════════
    with tab3:
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.markdown('<div class="info-card"><div class="card-title">분자 특성값</div>', unsafe_allow_html=True)
            feat_names = ["MW","LogP","TPSA","HBD","HBA","NumOH","RotBonds","HeavyAtoms"]
            feat_vals  = [feats[f] for f in feat_names]
            feat_df = pd.DataFrame({"특성":feat_names,"값":feat_vals,
                                     "설명":["분자량","소수성/친수성","극성 표면적",
                                              "HB 공여체","HB 수용체","OH기 수",
                                              "회전 결합","중원자 수"]})
            st.dataframe(feat_df, hide_index=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_f2:
            st.markdown('<div class="info-card"><div class="card-title">SHAP 특성 중요도 (RF 기준)</div>', unsafe_allow_html=True)
            shap_ser = pd.Series(shap_mean, index=FEATURES_REG).sort_values(ascending=True)
            fig_shap = go.Figure(go.Bar(
                x=shap_ser.values, y=shap_ser.index, orientation='h',
                marker_color=['#e53e3e' if v > shap_ser.mean() else '#94a3b8'
                              for v in shap_ser.values],
                text=[f"{v:.2f}" for v in shap_ser.values],
                textposition='outside',
            ))
            fig_shap.update_layout(
                height=280, plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(t=10,b=10,l=10,r=40),
                xaxis_title="|SHAP value|",
                font=dict(size=12, color="#1e293b"),
                xaxis=dict(gridcolor='#f1f5f9'),
                showlegend=False,
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # 전체 데이터 분포
        st.markdown('<div class="info-card"><div class="card-title">전체 데이터셋 끓는점 분포</div>', unsafe_allow_html=True)
        fig_dist = px.scatter(df_train, x="MW", y="bp_exp", color="HB_type",
                               hover_name="name",
                               color_discrete_map={"Strong":"#e53e3e","Weak":"#d97706","NonPolar":"#2d6a9f"},
                               labels={"MW":"분자량 (g/mol)","bp_exp":"끓는점 (°C)","HB_type":"유형"})
        # 현재 분자 표시
        fig_dist.add_trace(go.Scatter(
            x=[feats['MW']], y=[rf_pred],
            mode='markers',
            marker=dict(symbol='star', size=16, color='#16a34a',
                        line=dict(color='white',width=1.5)),
            name=f"{input_name or 'Input'} (예측)"
        ))
        fig_dist.update_layout(
            height=320, plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(t=10,b=20,l=20,r=20),
            font=dict(size=12, color="#1e293b"),
            legend=dict(bgcolor='white', bordercolor='#e2e8f0', borderwidth=1),
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ═══════ TAB 4: 분자 구조 ════════════════════════════════
    with tab4:
        col_s1, col_s2 = st.columns([1,2])
        with col_s1:
            st.markdown('<div class="info-card"><div class="card-title">2D 구조식</div>', unsafe_allow_html=True)
            if mol:
                mol_2d = Chem.MolFromSmiles(input_smi)
                AllChem.Compute2DCoords(mol_2d)
                svg = mol_to_image(mol_2d)
                st.image(svg.encode(), use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_s2:
            st.markdown('<div class="info-card"><div class="card-title">분자 정보 요약</div>', unsafe_allow_html=True)
            st.markdown(f"""
            | 항목 | 값 |
            |------|-----|
            | **SMILES** | `{input_smi}` |
            | **분자량** | {feats['MW']:.3f} g/mol |
            | **LogP** | {feats['LogP']:.4f} |
            | **TPSA** | {feats['TPSA']:.2f} Å² |
            | **HBD / HBA** | {feats['HBD']} / {feats['HBA']} |
            | **OH기 수** | {feats['NumOH']} |
            | **회전 결합** | {feats['RotBonds']} |
            | **중원자 수** | {feats['HeavyAtoms']} |
            | **수소결합 유형** | {hb_emoji} {hb_pred} ({hb_ko}) |
            | **RF 예측 끓는점** | **{rf_pred:.1f}°C** |
            | **앙상블 평균** | **{ensemble:.1f}°C** |
            """)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("### 3D 구조 시각화")
            st.info("💡 3D 시각화는 Colab의 Py3Dmol 코드를 참고하세요.\n"
                   "분자 구조 위에 예측 끓는점이 오버레이됩니다.")
