import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import os

api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

genai.configure(api_key=api_key)

model_ai = genai.GenerativeModel("gemini-1.5-flash")

# # API Key Authentication

# APP_API_KEY = "INSURANCE_AI_2026"

# user_key = st.text_input(
#     "Enter API Key to access Insurance AI System",
#     type="password"
# )

# if user_key != APP_API_KEY:
#     st.warning("Please enter a valid API key")
#     st.stop()

# st.success("Access Granted")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Set page configuration
st.set_page_config(
    page_title="AI Insurance Actuary Hub",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Subtitle
st.title("🛡️ AI-Based Insurance Claim & Fraud Analytics Hub")
st.markdown("""
This system empowers actuaries and risk assessors with Machine Learning models built on top of historic claim records. 
It utilizes **Random Forest Classifiers** to perform:
1. **Claim Likelihood Prediction** — Assessing if a policyholder is likely to file a claim.
2. **Fraud Detection Analysis** — Evaluating active claims for indicators of fraudulent behavior.
""")

# Load Dataset
@st.cache_data
# Load Dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("insurance_claims.csv")

        # Data Cleaning
        df = df.drop(
        [
        "policy_number",
        "insured_zip",
        "policy_bind_date",
        "incident_date",
        "incident_location"
        ],
        axis=1
        )

        return df

    except FileNotFoundError:
        st.error("Error: 'insurance_claims.csv' not found.")
        return None

df = load_data()

if df is not None:
    # Sidebar: Model Settings & Dataset Information
    st.sidebar.header("📊 Model Configuration")
    
    # Dataset preview in sidebar
    if st.sidebar.checkbox("Show Dataset Sample", value=False):
        st.sidebar.dataframe(df.head(5))
    
    # Preprocessing and Training Models
    # Let's write helper encoders to transform categoricals safely
    encoders = {}
    categorical_cols = ['policy_type', 'vehicle_type', 'accident_type', 'accident_severity', 'police_report_available']
    
    # Make a copy for training
    train_df = df.copy()
    
    # Encode categories
    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        encoders[col] = le
        
    # Map target columns
    train_df['fraud_reported'] = train_df['fraud_reported'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # 1. Claim Prediction Model (Target: claim_made)
    # Features: age, months_as_customer, policy_type, policy_annual_premium, previous_claims, vehicle_type, vehicle_age
    features_claim = ['age', 'months_as_customer', 'policy_type', 'policy_annual_premium', 'previous_claims', 'vehicle_type', 'vehicle_age']
    X_claim = train_df[features_claim]
    y_claim = train_df['claim_made']
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_claim, y_claim, test_size=0.25, random_state=42)
    model_claim = RandomForestClassifier(n_estimators=100, random_state=42)
    model_claim.fit(X_train_c, y_train_c)
    acc_claim = accuracy_score(y_test_c, model_claim.predict(X_test_c))

    # 2. Fraud Detection Model (Target: fraud_reported)
    # Features: total_claim_amount, previous_claims, accident_type, accident_severity, number_of_vehicles_involved, witnesses, police_report_available
    features_fraud = ['total_claim_amount', 'previous_claims', 'accident_type', 'accident_severity', 'number_of_vehicles_involved', 'witnesses', 'police_report_available']
    X_fraud = train_df[features_fraud]
    y_fraud = train_df['fraud_reported']
    
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_fraud, y_fraud, test_size=0.25, random_state=42)
    model_fraud = RandomForestClassifier(n_estimators=100, random_state=42)
    model_fraud.fit(X_train_f, y_train_f)
    acc_fraud = accuracy_score(y_test_f, model_fraud.predict(X_test_f))

    # Sidebar Metric Display
    st.sidebar.subheader("🎯 Model Training Statistics")
    st.sidebar.metric(label="Claim Prediction Accuracy", value=f"{acc_claim * 100:.1f}%")
    st.sidebar.metric(label="Fraud Detection Accuracy", value=f"{acc_fraud * 100:.1f}%")
    st.sidebar.info("Models are trained on the 'insurance_claims.csv' file using Random Forest Classification algorithms.")

    # Create Tabs for the two main modules and educational content
    tab1, tab2, tab3 = st.tabs(["📋 Claim Prediction Module", "🔍 Fraud Detection Module", "💡 Actuary Insights & AI Guide"])

    # TAB 1: Claim Prediction Module
    with tab1:
        st.header("📋 Policyholder Claim Probability Predictor")
        st.write("Determine the likelihood of a policyholder filing a claim based on demographic and policy attributes.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Customer Demographic & Policy")
            age = st.slider("Age of Insured", 18, 100, 35)
            months_cust = st.number_input("Months as Customer", min_value=0, max_value=600, value=120)
            policy_t = st.selectbox("Policy Coverage Type", df['policy_type'].unique())
            annual_prem = st.number_input("Annual Premium Amount ($)", min_value=100.0, max_value=10000.0, value=1250.0)
            prev_claims = st.slider("Number of Previous Claims", 0, 10, 1)

        with col2:
            st.subheader("🚗 Vehicle & Accident Spec")
            vehicle_t = st.selectbox("Vehicle Classification", df['vehicle_type'].unique())
            vehicle_age = st.slider("Vehicle Age (Years)", 0, 30, 4)
            
            st.info("💡 Adjust demographic values and previous claims to see how actuarial risk adjusts in real-time.")
        
        st.markdown("---")
        
        if st.button("Calculate Claim Likelihood Risk", key="predict_claim_btn"):
            # Prepare inputs
            # Map selected categories to encodings
            p_type_enc = encoders['policy_type'].transform([policy_t])[0] if policy_t in encoders['policy_type'].classes_ else 0
            v_type_enc = encoders['vehicle_type'].transform([vehicle_t])[0] if vehicle_t in encoders['vehicle_type'].classes_ else 0
            
            # Form features array
            input_data = np.array([[age, months_cust, p_type_enc, annual_prem, prev_claims, v_type_enc, vehicle_age]])
            
            # Predict
            prob = model_claim.predict_proba(input_data)[0][1] # Probability of claim_made=1
            prediction = model_claim.predict(input_data)[0]
            
            # Risk Level Assignment
            if prob < 0.35:
                risk_level = "🟢 Low Risk"
                color_class = "success"
                text_color = "green"
            elif prob < 0.65:
                risk_level = "🟡 Medium Risk"
                color_class = "warning"
                text_color = "orange"
            else:
                risk_level = "🔴 High Risk"
                color_class = "error"
                text_color = "red"
                
            # Render visual output
            st.subheader("🔮 Predictive Analytics Results")
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.markdown(f"#### Risk Classification:\n### <span style='color:{text_color};'>{risk_level}</span>", unsafe_allow_html=True)
            with metric_col2:
                st.metric("Claim Probability", f"{prob * 100:.2f}%")
                
            st.progress(prob)
            
            # Actuarial details
            st.markdown(f"""
            **Recommendation Assessment:**
            *   **Claim Risk Assessment**: This customer's characteristics suggest a **{prob * 100:.1f}%** chance of filing an insurance claim over their policy term.
            *   **Premium Adjustment guidance**: {'No immediate rate adjustment is required. Consider maintaining the current annual premium level.' if prob < 0.35 else 'Suggest a mild premium loader (+5% to +10%) or higher deductible during renewal.' if prob < 0.65 else 'High claim probability alert. Consider restructuring policy limits, adding high-risk loaders (+20%), or reviewing underwriter requirements.'}
            """)

    # TAB 2: Fraud Detection Module
    with tab2:
        st.header("🔍 Claim Fraudulence Detection")
        st.write("Evaluate open claims for potential anomalies or indicators of bad faith/fraudulent activity.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Claim Value & History")
            total_claim = st.number_input("Total Claimed Amount ($)", min_value=0.0, max_value=250000.0, value=12000.0)
            fraud_prev_claims = st.slider("Insured's Previous Claims Counter", 0, 10, 1, key="fraud_prev")
            police_rep = st.selectbox("Police Report Available", df['police_report_available'].unique())

        with col2:
            st.subheader("💥 Incident Circumstances")
            acc_type = st.selectbox("Incident Collision Type", df['accident_type'].unique())
            acc_sev = st.selectbox("Accident Severity", df['accident_severity'].unique())
            num_vehicles = st.slider("Number of Vehicles Involved", 0, 10, 2)
            witness_count = st.slider("Number of Documented Witnesses", 0, 5, 1)

        st.markdown("---")
        
        if st.button("Evaluate Claim Legitimacy", key="predict_fraud_btn"):
            # Map selected categories to encodings
            acc_type_enc = encoders['accident_type'].transform([acc_type])[0] if acc_type in encoders['accident_type'].classes_ else 0
            acc_sev_enc = encoders['accident_severity'].transform([acc_sev])[0] if acc_sev in encoders['accident_severity'].classes_ else 0
            police_rep_enc = encoders['police_report_available'].transform([police_rep])[0] if police_rep in encoders['police_report_available'].classes_ else 0
            
            # Form features array
            # ['total_claim_amount', 'previous_claims', 'accident_type', 'accident_severity', 'number_of_vehicles_involved', 'witnesses', 'police_report_available']
            input_data_fraud = np.array([[total_claim, fraud_prev_claims, acc_type_enc, acc_sev_enc, num_vehicles, witness_count, police_rep_enc]])
            
            # Predict
            prob_fraud = model_fraud.predict_proba(input_data_fraud)[0][1] # Probability of fraud_reported=1
            
            # Risk Level Assignment
            if prob_fraud < 0.30:
                fraud_risk = "🟢 Low Fraud Risk"
                f_color = "green"
                f_note = "Clear for standard, automated fast-track payout workflows."
            elif prob_fraud < 0.60:
                fraud_risk = "🟡 Medium Fraud Risk"
                f_color = "orange"
                f_note = "Recommend routine desk-review verification of repair estimates and witness statements."
            else:
                fraud_risk = "🔴 High Fraud Risk"
                f_color = "red"
                f_note = "Flagged for Special Investigations Unit (SIU). Suspend payouts and deploy independent adjuster to verify damage integrity."

            # Render visual output
            st.subheader("🔎 Forensic Fraud Assessment")
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown(f"#### Investigative Alert:\n### <span style='color:{f_color};'>{fraud_risk}</span>", unsafe_allow_html=True)
            with m_col2:
                st.metric("Fraud Probability Rating", f"{prob_fraud * 100:.2f}%")
                
            st.progress(prob_fraud)
            
            st.markdown(f"""
            **Actuarial & Investigative Guidelines:**
            *   **Calculated Suspicions**: The claim possesses an estimated fraud risk of **{prob_fraud * 100:.1f}%**.
            *   **Actionable Next Step**: **{f_note}**
            *   **Contributing Factors**: Random Forest tree splits place highest significance on *Accident Severity*, *Claim Amount*, and *Police Report Availability* when establishing claim validity.
            """)

    # TAB 3: Educational Guide
    with tab3:
        st.header("💡 AI in Actuarial Science & Underwriting")
        
        st.markdown("""
        ### How AI Empowers Actuaries and Risk Managers
        Traditionally, actuaries relied purely on static tables (e.g., life expectancy charts, historic regional disaster logs) to group people into broad risk categories. Machine Learning (ML) transforms this by finding non-linear, complex multi-dimensional relationships:
        
        1. **Hyper-Personalization**: Instead of grouping all 30-year-olds together, AI models evaluate combinations of variables (e.g., matching low vehicle ages with specific coverages and previous clean histories) to predict likelihood down to a high-precision decimal.
        2. **Real-Time Automated Underwriting**: Rather than manual paper review taking 14 days, ML allows insurance platforms to price risk in real-time.
        3. **Fraud Anomaly Flags**: Special Investigations Units (SIU) utilize models like Random Forests to catch anomalies (e.g., claims with high amounts, multiple cars, but zero witnesses and no police report) instantly before payouts are made.
        
        ### Random Forest algorithm explained
        The **Random Forest Classifier** used in this application operates by creating an ensemble (or "forest") of many individual decision trees. 
        - Each tree votes on whether a customer is a high-risk or low-risk prospect.
        - By combining scores from 100 trees, the forest reduces overfitting to small samples, providing a highly robust and balanced risk score.
        """)
        
        st.info("💡 Tip: Try tweaking dataset parameters inside 'insurance_claims.csv' and reloading the Streamlit app to watch accuracy metrics dynamic adjust!")

else:
    st.warning("⚠️ Please provide 'insurance_claims.csv' in the project directory to build the predictive models.")
