"""
All-In-One Streamlit App: Trains on Startup & Predicts
Upgraded Logistic Regression Pipeline with SMOTE for Extreme Class Imbalance.

Requirements:
    pip install streamlit pandas numpy scipy scikit-learn imbalanced-learn openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦", layout="centered")

MAX_ITERATIONS = 10

# ─────────────────────────────────────────────────────────────────────────────
# 1. LIVE TRAINING ENGINE (Runs on site opening)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Applying SMOTE and training model...")
def train_model(max_iter):
    try:
        # Load Data
        df = pd.read_excel('Case_study_data.xlsx').dropna()
        y  = df['default'].values.astype(float)

        # Define Variables
        XCOLS = [
            'ms1','ms2','ms3','ms4','ms6',
            'rs1','rs2','rs3','rs4',
            'es1','es2','es3','es4',
            'a2','a4','a6',
            'lmicr','lmice','ltinc','lpincb',
            'lltv','lapr','lterm',
            'lcarprice','lcarage',
        ]
        NAMES = ['const'] + XCOLS

        X = df[XCOLS].values.astype(float)
        
        # 1. TRAIN/TEST SPLIT (80% Train, 20% Test)
        # stratify=y ensures the 40 defaults are split perfectly 80/20 (32 in train, 8 in test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 2. APPLY SMOTE (Synthetic Minority Over-sampling Technique)
        # This creates synthetic 'default' cases so the model actually learns their patterns
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # 3. TRAIN THE MODEL on the balanced data
        clf = LogisticRegression(
            penalty='l2',          
            C=1.0,                 
            max_iter=max_iter, 
            solver='lbfgs'
        )
        clf.fit(X_train_balanced, y_train_balanced)
        
        # 4. EVALUATE on the UNTOUCHED Test Set
        train_probs = clf.predict_proba(X_train_balanced)[:, 1]
        test_probs  = clf.predict_proba(X_test)[:, 1]
        
        train_auc = roc_auc_score(y_train_balanced, train_probs)
        test_auc  = roc_auc_score(y_test, test_probs)
        
        # Extract coefficients and intercept
        beta = np.concatenate(([clf.intercept_[0]], clf.coef_[0]))
        converged_iter = clf.n_iter_[0]
                
        # Return weights, iterations, and metrics
        metrics = {
            'train_auc': train_auc, 
            'test_auc': test_auc,
            'original_train_size': len(y_train),
            'smote_train_size': len(y_train_balanced)
        }
        return dict(zip(NAMES, beta)), converged_iter, metrics
        
    except FileNotFoundError:
        return None, 0, None
    except Exception as e:
        st.error(f"Training error: {e}")
        return None, 0, None

# Trigger the training function
coefs, iters_taken, metrics = train_model(max_iter=MAX_ITERATIONS)

if coefs is None:
    st.error("🚨 `Case_study_data.xlsx` not found! Please place it in the same folder as this script.")
    st.stop()

# Display Performance Metrics in the Sidebar
st.sidebar.markdown("### 📊 Model Diagnostics")
st.sidebar.success(f"✅ Trained in {iters_taken} iterations.")
st.sidebar.info(f"**SMOTE Applied:** Expanded training data from {metrics['original_train_size']} rows to {metrics['smote_train_size']} rows to balance classes.")
st.sidebar.markdown("**(ROC-AUC > 0.70 is good)**")
st.sidebar.metric(label="Train ROC-AUC", value=f"{metrics['train_auc']:.3f}")
st.sidebar.metric(label="Test ROC-AUC", value=f"{metrics['test_auc']:.3f}")

# Overfit/Underfit Analysis Logic
if metrics['train_auc'] - metrics['test_auc'] > 0.10:
    st.sidebar.warning("⚠️ High gap between Train/Test. Model is slightly **Overfitting**.")
elif metrics['train_auc'] < 0.65:
    st.sidebar.error("⚠️ Low scores overall. Model is **Underfitting**. A simple linear model might not be powerful enough for this dataset.")
else:
    st.sidebar.success("🎯 Train and Test scores are solid. The model handles the imbalance well!")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PREDICTION UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏦 Subprime Loan Default Predictor")

input_mode = st.radio(
    "How are you entering the numerical data?", 
    ["Raw Values (App will calculate Logs automatically)", "Log Values (I am entering the exact Logs)"]
)
is_raw = "Raw" in input_mode

with st.form("prediction_form"):
    st.markdown("### 👤 Personal & Financial Details")
    col1, col2 = st.columns(2)
    with col1:
        micr_val = st.number_input("Monthly Income", value=4000.0 if is_raw else 8.29)
        mice_val = st.number_input("Monthly Expenses", value=2500.0 if is_raw else 7.82)
    with col2:
        tinc_val = st.number_input("Total Income", value=4500.0 if is_raw else 8.41)
        pincb_val = st.number_input("Personal Income", value=4000.0 if is_raw else 8.29)

    st.markdown("### 🚙 Vehicle & Loan Details")
    col3, col4 = st.columns(2)
    with col3:
        carprice_val = st.number_input("Car Price", value=12000.0 if is_raw else 9.39)
        carage_val = st.number_input("Car Age (Years)", value=5.0 if is_raw else 1.60)
    with col4:
        ltv_val = st.number_input("Loan-to-Value (e.g. 0.95 or 95)", value=0.95 if is_raw else -0.05)
        apr_val = st.number_input("APR (e.g. 15 for 15%)", value=15.0 if is_raw else 2.70)
        term_val = st.number_input("Term length (Months)", value=60.0 if is_raw else 4.09)

    st.markdown("### 📋 Categorical Indicators (Check if applicable)")
    cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)
    with cat_col1:
        st.markdown("**Marital Status**")
        ms1 = st.checkbox("ms1"); ms2 = st.checkbox("ms2")
        ms3 = st.checkbox("ms3"); ms4 = st.checkbox("ms4"); ms6 = st.checkbox("ms6")
    with cat_col2:
        st.markdown("**Residential**")
        rs1 = st.checkbox("rs1"); rs2 = st.checkbox("rs2")
        rs3 = st.checkbox("rs3"); rs4 = st.checkbox("rs4")
    with cat_col3:
        st.markdown("**Employment**")
        es1 = st.checkbox("es1"); es2 = st.checkbox("es2")
        es3 = st.checkbox("es3"); es4 = st.checkbox("es4")
    with cat_col4:
        st.markdown("**Geographic Area**")
        a2 = st.checkbox("a2"); a4 = st.checkbox("a4"); a6 = st.checkbox("a6")

    submitted = st.form_submit_button("🔮 Predict Default Probability", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. PREDICTION CALCULATION
# ─────────────────────────────────────────────────────────────────────────────
if submitted:
    def safe_log(x):
        return np.log(x) if x > 0 else 0.0

    if is_raw:
        user_inputs = {
            'lmicr': safe_log(micr_val), 'lmice': safe_log(mice_val), 
            'ltinc': safe_log(tinc_val), 'lpincb': safe_log(pincb_val),
            'lcarprice': safe_log(carprice_val), 'lcarage': safe_log(carage_val),
            'lltv': safe_log(ltv_val), 'lapr': safe_log(apr_val), 'lterm': safe_log(term_val),
        }
    else:
        user_inputs = {
            'lmicr': micr_val, 'lmice': mice_val, 
            'ltinc': tinc_val, 'lpincb': pincb_val,
            'lcarprice': carprice_val, 'lcarage': carage_val,
            'lltv': ltv_val, 'lapr': apr_val, 'lterm': term_val,
        }

    user_inputs.update({
        'ms1': int(ms1), 'ms2': int(ms2), 'ms3': int(ms3), 'ms4': int(ms4), 'ms6': int(ms6),
        'rs1': int(rs1), 'rs2': int(rs2), 'rs3': int(rs3), 'rs4': int(rs4),
        'es1': int(es1), 'es2': int(es2), 'es3': int(es3), 'es4': int(es4),
        'a2': int(a2), 'a4': int(a4), 'a6': int(a6)
    })

    base_const = coefs.get('const', 0.0)
    linear_predictor = base_const
    
    debug_data = [{'Variable': 'Intercept (const)', 'Final Value Used': 1.0, 'Coefficient': base_const, 'Contribution': base_const}]

    for var_name, value in user_inputs.items():
        if var_name in coefs:
            coef = coefs[var_name]
            contribution = coef * value
            linear_predictor += contribution
            
            if value != 0:
                debug_data.append({
                    'Variable': var_name, 
                    'Final Value Used': round(value, 4), 
                    'Coefficient': round(coef, 6), 
                    'Contribution': round(contribution, 4)
                })

    probability = expit(linear_predictor)
    
    st.divider()
    
    # Because SMOTE balances the training data to 50/50, the output probability 
    # is now calibrated to a 50/50 baseline. A 50% probability means the model is truly torn.
    if probability < 0.40:
        risk_color = "green"; risk_label = "Low Risk"
    elif probability <= 0.60:
        risk_color = "orange"; risk_label = "Moderate Risk"
    else:
        risk_color = "red"; risk_label = "High Risk"

    st.markdown(f"""
        <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.05); border: 2px solid {risk_color};">
            <h2 style="margin:0;">Probability of Default</h2>
            <h1 style="color: {risk_color}; font-size: 4rem; margin: 10px 0;">{probability * 100:.2f}%</h1>
            <h3 style="margin:0; color: {risk_color};">{risk_label}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🛠️ View Math Breakdown"):
        st.write(f"**Total Linear Sum ($X \\beta$):** `{linear_predictor:.4f}`")
        st.latex(r"P(Default) = \frac{1}{1 + e^{-X\beta}}")
        st.dataframe(pd.DataFrame(debug_data), use_container_width=True)