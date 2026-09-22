# ============================================================
# app/app.py — Accounting Fraud Detection (Streamlit)
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_shap import st_shap
import shap

# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Accounting Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 2. CUSTOM CSS FOR MODERN UI
# ============================================================
st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; }
    .sub-title  { font-size: 1rem; color: #6B7280; margin-bottom: 2rem; }
    .metric-card {
        background: #F9FAFB; border: 1px solid #E5E7EB;
        border-radius: 12px; padding: 1rem; text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #111827; }
    .metric-label { font-size: 0.85rem; color: #6B7280; }
    .stButton>button {
        background: #1E3A8A; color: white; border-radius: 8px;
        padding: 0.6rem 1.2rem; font-weight: 600; border: none;
    }
    .stButton>button:hover { background: #1E40AF; }
    .fraud-box {
        background: #FEF2F2; border: 2px solid #EF4444;
        border-radius: 12px; padding: 1.5rem; text-align: center;
    }
    .safe-box {
        background: #F0FDF4; border: 2px solid #22C55E;
        border-radius: 12px; padding: 1.5rem; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. LOAD MODEL (CACHED)
# ============================================================
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base, 'models', 'final_xgboost_model')

    pipeline = joblib.load(os.path.join(model_dir, 'pipeline.pkl'))
    with open(os.path.join(model_dir, 'threshold.json')) as f:
        threshold = json.load(f)['optimal_threshold']
    with open(os.path.join(model_dir, 'features.json')) as f:
        features = json.load(f)

    return pipeline, threshold, features

PIPELINE, DEFAULT_THRESHOLD, FEATURES = load_artifacts()

# ============================================================
# 4. FEATURE ALIASES
# ============================================================
FEATURE_ALIASES = {
    'total assets': 'at', 'assets': 'at',
    'total liabilities': 'lt', 'liabilities': 'lt',
    'current assets': 'act',
    'current liabilities': 'lct',
    'net income': 'ni', 'net income loss': 'ni',
    'net sales': 'sale', 'sales': 'sale', 'revenue': 'sale', 'turnover': 'sale',
    'common equity': 'ceq', 'equity': 'ceq', 'ordinary equity': 'ceq',
    'cash': 'che', 'cash and short term investments': 'che',
    'inventories': 'invt', 'inventory': 'invt',
    'receivables': 'rect', 'accounts receivable': 'rect', 'total receivables': 'rect',
    'retained earnings': 're',
    'cost of goods sold': 'cogs', 'cogs': 'cogs',
    'accounts payable': 'ap', 'trade payables': 'ap',
    'property plant and equipment': 'ppegt', 'ppe': 'ppegt',
    'long term debt': 'dltt', 'debt in current liabilities': 'dlc',
    'depreciation': 'dp', 'depreciation and amortization': 'dp',
    'income before extraordinary items': 'ib',
    'income taxes': 'txt', 'income taxes payable': 'txp',
    'interest expense': 'xint', 'common shares outstanding': 'csho',
    'preferred stock': 'pstk', 'stock issuance': 'sstk',
    'long term debt issuance': 'dltis',
    'price close': 'prcc_f', 'share price': 'prcc_f',
    'soft assets': 'soft_assets',
    'retained earnings over total assets': 'reoa',
    'earnings before interest and taxes': 'EBIT',
    'book to market': 'bm', 'book to market ratio': 'bm',
    'depreciation index': 'dpi',
    'change in receivables': 'dch_rec',
    'change in inventory': 'dch_inv',
    'change in return on assets': 'ch_roa',
    'change in free cash flow': 'ch_fcf',
    'change in cash sales': 'ch_cs',
    'change in cash margin': 'ch_cm',
    'working capital accruals': 'dch_wc',
    'rsst accruals': 'ch_rsst',
    'actual issuance': 'issue',
}

def normalize_columns(df):
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in FEATURE_ALIASES:
            renamed[col] = FEATURE_ALIASES[key]
        elif col in FEATURES:
            renamed[col] = col
    return df.rename(columns=renamed)

# ============================================================
# 5. AUTO-CALCULATE RATIOS FROM RAW INPUTS
# ============================================================
def calculate_ratios(raw):
    """Compute 10 ratios from raw inputs. Returns dict."""
    r = {}
    eps = 1e-6

    at = raw.get('at', 0)
    lt = raw.get('lt', 0)
    act = raw.get('act', 0)
    lct = raw.get('lct', 0)
    ceq = raw.get('ceq', 0)
    che = raw.get('che', 0)
    invt = raw.get('invt', 0)
    rect = raw.get('rect', 0)
    re = raw.get('re', 0)
    ni = raw.get('ni', 0)
    sale = raw.get('sale', 0)
    cogs = raw.get('cogs', 0)

    if at + eps != 0:
        r['debt_to_assets'] = lt / (at + eps)
        r['roa'] = ni / (at + eps)
        r['equity_ratio'] = ceq / (at + eps)
        r['cash_ratio'] = che / (at + eps)
        r['retained_earnings_ratio'] = re / (at + eps)
    if lct + eps != 0:
        r['current_ratio'] = act / (lct + eps)
        r['quick_ratio'] = (che + rect) / (lct + eps)
    if sale + eps != 0:
        r['profit_margin'] = ni / (sale + eps)
    if rect + eps != 0:
        r['receivables_turnover'] = sale / (rect + eps)
    if invt + eps != 0:
        r['inventory_turnover'] = cogs / (invt + eps)

    return r

def validate_ratios(calculated, user_provided):
    """Compare calculated vs user-provided ratios. Returns list of discrepancies."""
    discrepancies = []
    tolerance = 0.01  # 1% tolerance
    for ratio, calc_val in calculated.items():
        if ratio in user_provided and user_provided[ratio] != 0:
            user_val = user_provided[ratio]
            if abs(calc_val - user_val) > tolerance * max(abs(calc_val), abs(user_val), 1):
                discrepancies.append({
                    'ratio': ratio,
                    'calculated': calc_val,
                    'provided': user_val
                })
    return discrepancies

# ============================================================
# 6. SHAP EXPLAINER (CACHED)
# ============================================================
@st.cache_resource
def get_shap_explainer():
    classifier = PIPELINE.named_steps['classifier']
    return shap.TreeExplainer(classifier)

SHAP_EXPLAINER = get_shap_explainer()

def get_shap_values(df):
    """Compute SHAP values for a dataframe of raw features."""
    X = df[FEATURES].copy()
    preprocessor = PIPELINE.named_steps['preprocessor']
    X_pre = preprocessor.transform(X)
    shap_values = SHAP_EXPLAINER.shap_values(X_pre)
    return shap_values, X_pre

# ============================================================
# 7. PREDICTION
# ============================================================
def run_prediction(df):
    for col in FEATURES:
        if col not in df.columns:
            df[col] = float('nan')
    X = df[FEATURES].copy()
    proba = PIPELINE.predict_proba(X)[:, 1]
    return proba

# ============================================================
# 8. HEADER
# ============================================================
st.markdown('<div class="main-title">🔍 Accounting Fraud Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Machine learning model to detect accounting fraud from financial statement data</div>', unsafe_allow_html=True)

# ============================================================
# 9. TABS
# ============================================================
tab_manual, tab_upload = st.tabs(["✍️ Manual Entry", "📁 Upload File"])

# ------------------------------------------------------------
# TAB 1: MANUAL ENTRY
# ------------------------------------------------------------
with tab_manual:
    st.markdown("### Enter financial values for a single firm-year")
    st.caption("All values must be non-negative. Ratios are auto-calculated and validated.")

    # Threshold slider
    threshold = st.slider(
        "Fraud Probability Threshold",
        min_value=0.1, max_value=0.95,
        value=float(DEFAULT_THRESHOLD), step=0.05,
        help="Lower threshold = more fraud flagged (higher recall). Higher threshold = fewer false alarms (higher precision)."
    )

    # Organize inputs
    with st.expander("📊 Balance Sheet", expanded=True):
        c1, c2, c3 = st.columns(3)
        at = c1.number_input("Total Assets (at)", min_value=0.0, value=0.0, format="%.4f")
        lt = c2.number_input("Total Liabilities (lt)", min_value=0.0, value=0.0, format="%.4f")
        act = c3.number_input("Current Assets (act)", min_value=0.0, value=0.0, format="%.4f")
        lct = c1.number_input("Current Liabilities (lct)", min_value=0.0, value=0.0, format="%.4f")
        ceq = c2.number_input("Common Equity (ceq)", min_value=0.0, value=0.0, format="%.4f")
        che = c3.number_input("Cash & ST Investments (che)", min_value=0.0, value=0.0, format="%.4f")
        invt = c1.number_input("Inventories (invt)", min_value=0.0, value=0.0, format="%.4f")
        rect = c2.number_input("Receivables (rect)", min_value=0.0, value=0.0, format="%.4f")
        re = c3.number_input("Retained Earnings (re)", min_value=0.0, value=0.0, format="%.4f")
        ppegt = c1.number_input("Property Plant & Equip (ppegt)", min_value=0.0, value=0.0, format="%.4f")
        ap = c2.number_input("Accounts Payable (ap)", min_value=0.0, value=0.0, format="%.4f")

    with st.expander("📈 Income Statement", expanded=False):
        c1, c2, c3 = st.columns(3)
        sale = c1.number_input("Sales/Turnover (sale)", min_value=0.0, value=0.0, format="%.4f")
        cogs = c2.number_input("Cost of Goods Sold (cogs)", min_value=0.0, value=0.0, format="%.4f")
        ni = c3.number_input("Net Income (ni)", min_value=0.0, value=0.0, format="%.4f")
        ib = c1.number_input("Income Before Extraordinary (ib)", min_value=0.0, value=0.0, format="%.4f")
        dp = c2.number_input("Depreciation (dp)", min_value=0.0, value=0.0, format="%.4f")
        txt = c3.number_input("Income Taxes (txt)", min_value=0.0, value=0.0, format="%.4f")
        txp = c1.number_input("Income Taxes Payable (txp)", min_value=0.0, value=0.0, format="%.4f")
        xint = c2.number_input("Interest Expense (xint)", min_value=0.0, value=0.0, format="%.4f")

    with st.expander("💰 Equity & Debt", expanded=False):
        c1, c2, c3 = st.columns(3)
        csho = c1.number_input("Common Shares Outstanding (csho)", min_value=0.0, value=0.0, format="%.4f")
        pstk = c2.number_input("Preferred Stock (pstk)", min_value=0.0, value=0.0, format="%.4f")
        sstk = c3.number_input("Stock Issuance (sstk)", min_value=0.0, value=0.0, format="%.4f")
        dlc = c1.number_input("Debt in Current Liab (dlc)", min_value=0.0, value=0.0, format="%.4f")
        dltt = c2.number_input("Long-Term Debt (dltt)", min_value=0.0, value=0.0, format="%.4f")
        dltis = c3.number_input("LT Debt Issuance (dltis)", min_value=0.0, value=0.0, format="%.4f")
        prcc_f = c1.number_input("Share Price (prcc_f)", min_value=0.0, value=0.0, format="%.4f")

    with st.expander("🧮 Derived Financial Ratios (Optional — auto-calculated if left at 0)", expanded=False):
        st.caption("Leave at 0 to auto-calculate from raw values above. Provide values to validate against calculation.")
        c1, c2, c3 = st.columns(3)
        soft_assets = c1.number_input("Soft Assets (soft_assets)", min_value=0.0, value=0.0, format="%.4f")
        reoa = c2.number_input("Retained Earnings/Assets (reoa)", min_value=0.0, value=0.0, format="%.4f")
        EBIT = c3.number_input("EBIT / Assets", min_value=0.0, value=0.0, format="%.4f")
        bm = c1.number_input("Book-to-Market (bm)", min_value=0.0, value=0.0, format="%.4f")
        dpi = c2.number_input("Depreciation Index (dpi)", min_value=0.0, value=0.0, format="%.4f")
        dch_wc = c3.number_input("WC Accruals (dch_wc)", min_value=0.0, value=0.0, format="%.4f")
        ch_rsst = c1.number_input("RSST Accruals (ch_rsst)", min_value=0.0, value=0.0, format="%.4f")
        dch_rec = c2.number_input("Change in Receivables (dch_rec)", min_value=0.0, value=0.0, format="%.4f")
        dch_inv = c3.number_input("Change in Inventory (dch_inv)", min_value=0.0, value=0.0, format="%.4f")
        ch_cs = c1.number_input("Change in Cash Sales (ch_cs)", min_value=0.0, value=0.0, format="%.4f")
        ch_cm = c2.number_input("Change in Cash Margin (ch_cm)", min_value=0.0, value=0.0, format="%.4f")
        ch_roa = c3.number_input("Change in ROA (ch_roa)", min_value=0.0, value=0.0, format="%.4f")
        ch_fcf = c1.number_input("Change in FCF (ch_fcf)", min_value=0.0, value=0.0, format="%.4f")
        issue = c2.number_input("Actual Issuance (issue)", min_value=0.0, value=0.0, format="%.4f")

    # ---- Load Sample Fraud Button ----
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("📥 Load Sample Fraud"):
            st.session_state['manual_inputs'] = {
                'at': 32.335, 'lt': 26.073, 'act': 10.047, 'lct': 8.5,
                'ceq': 6.262, 'che': 0.002, 'invt': 1.2, 'rect': 2.5,
                're': 5.42, 'ppegt': 15.0, 'ap': 3.736, 'sale': 30.633,
                'cogs': 25.0, 'ni': -0.5, 'ib': -0.4, 'dp': 1.2,
                'txt': 0.3, 'txp': 0.1, 'xint': 0.8, 'csho': 1.0,
                'pstk': 0.0, 'sstk': 0.5, 'dlc': 1.5, 'dltt': 10.0,
                'dltis': 2.0, 'prcc_f': 5.0, 'soft_assets': 0.312,
                'reoa': 0.167, 'EBIT': 0.162, 'bm': 0.413, 'dpi': 0.874,
                'dch_wc': 0.05, 'ch_rsst': 0.08, 'dch_rec': 0.12,
                'dch_inv': -0.03, 'ch_cs': 0.095, 'ch_cm': 0.083,
                'ch_roa': -0.020, 'ch_fcf': -0.042, 'issue': 1.0,
            }
            st.rerun()

    # ---- Predict ----
    if st.button("🔮 Predict Fraud", key="predict_manual"):
        manual_dict = {
            'at': at, 'lt': lt, 'act': act, 'lct': lct, 'ceq': ceq,
            'che': che, 'invt': invt, 'rect': rect, 're': re,
            'ppegt': ppegt, 'ap': ap, 'sale': sale, 'cogs': cogs,
            'ni': ni, 'ib': ib, 'dp': dp, 'txt': txt, 'txp': txp,
            'xint': xint, 'csho': csho, 'pstk': pstk, 'sstk': sstk,
            'dlc': dlc, 'dltt': dltt, 'dltis': dltis, 'prcc_f': prcc_f,
            'soft_assets': soft_assets, 'reoa': reoa, 'EBIT': EBIT,
            'bm': bm, 'dpi': dpi, 'dch_wc': dch_wc, 'ch_rsst': ch_rsst,
            'dch_rec': dch_rec, 'dch_inv': dch_inv, 'ch_cs': ch_cs,
            'ch_cm': ch_cm, 'ch_roa': ch_roa, 'ch_fcf': ch_fcf,
            'issue': issue,
        }

        # Auto-calculate ratios from raw inputs
        calculated = calculate_ratios(manual_dict)

        # Validate user-provided ratios against calculated
        user_ratios = {k: v for k, v in manual_dict.items() if k in calculated and v != 0}
        discrepancies = validate_ratios(calculated, user_ratios)

        if discrepancies:
            st.warning("⚠️ Ratio discrepancies detected (calculated vs. entered):")
            for d in discrepancies:
                st.write(f"- **{d['ratio']}**: calculated = {d['calculated']:.4f}, entered = {d['provided']:.4f}")

        # Merge: use calculated ratios for model input (overrides manual entries)
        for k, v in calculated.items():
            manual_dict[k] = v

        manual_df = pd.DataFrame([manual_dict])
        proba = run_prediction(manual_df)
        pred = int(proba[0] >= threshold)

        # ---- Result Box ----
        st.markdown("### Prediction Result")
        if pred == 1:
            st.markdown(f'<div class="fraud-box"><h2>⚠️ FRAUD DETECTED</h2><p>Fraud Probability: <b>{proba[0]:.4f}</b></p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="safe-box"><h2>✅ NON-FRAUDULENT</h2><p>Fraud Probability: <b>{proba[0]:.4f}</b></p></div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Fraud Probability", f"{proba[0]:.4f}")
        m2.metric("Decision", "⚠️ FRAUD" if pred == 1 else "✅ NON-FRAUD")
        m3.metric("Threshold Used", f"{threshold:.4f}")

        # ---- Risk Gauge ----
        st.markdown("#### Risk Gauge")
        gauge_spec = {
            "series": [{
                "type": "gauge",
                "startAngle": 180,
                "endAngle": 0,
                "min": 0,
                "max": 1,
                "splitNumber": 5,
                "axisLine": {
                    "lineStyle": {
                        "width": 30,
                        "color": [[0.3, "#22C55E"], [0.6, "#F59E0B"], [1, "#EF4444"]]
                    }
                },
                "pointer": {"itemStyle": {"color": "#1E3A8A"}},
                "detail": {
                    "formatter": f"{proba[0]:.4f}",
                    "fontSize": 24,
                    "fontWeight": "bold",
                    "color": "#111827"
                },
                "data": [{"value": float(proba[0])}]
            }]
        }
        st.echarts_chart(gauge_spec, height=250)

        # ---- SHAP Waterfall ----
        st.markdown("#### Why this prediction? (SHAP Explanation)")
        try:
            shap_values, X_pre = get_shap_values(manual_df)
            # shap_values shape: (n_samples, n_features)
            # For binary classification, TreeExplainer returns a single array
            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=SHAP_EXPLAINER.expected_value,
                data=X_pre[0],
                feature_names=FEATURES
            )
            st_shap(shap.plots.waterfall(explanation), height=400)
        except Exception as e:
            st.info(f"SHAP plot unavailable: {e}")

# ------------------------------------------------------------
# TAB 2: FILE UPLOAD
# ------------------------------------------------------------
with tab_upload:
    st.markdown("### Upload a CSV or Excel file")
    st.caption("The file should contain financial statement variables. Column names are auto-mapped from full names to model abbreviations.")

    threshold_upload = st.slider(
        "Fraud Probability Threshold (Upload)",
        min_value=0.1, max_value=0.95,
        value=float(DEFAULT_THRESHOLD), step=0.05,
        key="threshold_upload"
    )

    uploaded_file = st.file_uploader(
        "Drop your file here",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            df = normalize_columns(df)

            st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

            with st.spinner("Analysing..."):
                proba = run_prediction(df)
                preds = (proba >= threshold_upload).astype(int)

            results = df.copy()
            results['fraud_probability'] = proba
            results['fraud_flag'] = preds

            n_total = len(preds)
            n_flagged = int(preds.sum())

            m1, m2, m3 = st.columns(3)
            m1.metric("Observations", n_total)
            m2.metric("Flagged as Fraud", n_flagged)
            m3.metric("Fraud Rate", f"{n_flagged/max(n_total,1):.2%}")

            st.markdown("### Results")
            st.dataframe(results[['fraud_probability', 'fraud_flag']], use_container_width=True)

            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Results",
                data=csv,
                file_name="fraud_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Please upload a file to begin.")