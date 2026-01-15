import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import io

# 1. AUTHENTICATION & PERSONA [cite: 52]
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

st.title("📊 DEPT Strategic Financial Engine")

# Normalization for "Smart" Name Matching [cite: 34]
def normalize(name):
    if pd.isna(name): return ""
    return str(name).lower().strip().replace(" ", "").replace("'", "")

# 2. DATA INGESTION (DEPTapps & Billing Support) [cite: 14, 43]
with st.sidebar:
    st.header("Upload Sources")
    # Supports Excel formats as discussed 
    hours_file = st.file_uploader("Upload DEPTapps Export", type=['xlsx', 'xls', 'csv'])
    pricing_file = st.file_uploader("Upload Pricing Plan (CPT)", type=['xlsx', 'xls', 'csv'])
    billing_file = st.file_uploader("Upload Billing Worksheet", type=['xlsx', 'xls', 'csv'])

if hours_file and pricing_file and billing_file:
    # Load data dynamically
    df_hours = pd.read_excel(hours_file) if hours_file.name.endswith(('xlsx', 'xls')) else pd.read_csv(hours_file)
    df_plan = pd.read_excel(pricing_file) if pricing_file.name.endswith(('xlsx', 'xls')) else pd.read_csv(pricing_file)
    df_bill = pd.read_excel(billing_file) if billing_file.name.endswith(('xlsx', 'xls')) else pd.read_csv(billing_file)

    # 3. SMART INTERPRETATION (THE "SMART" LAYER) [cite: 158, 179]
    # We feed a data preview to Gemini so it understands the 'Why' behind the numbers
    data_preview = f"""
    HOURS SAMPLE: {df_hours.head(10).to_string()}
    PLAN SAMPLE: {df_plan.head(10).to_string()}
    BILLING SAMPLE: {df_bill.head(10).to_string()}
    """
    
    st.info("🤖 Gemini is interpreting file structures and varied income...")
    
    # AI identifies the varied income and necessary columns
    intel_prompt = f"""
    Analyze these DEPT financial snippets. 
    1. Identify the 'Dept Fee' or 'Income' that varies with media spend. 
    2. Identify which column in 'HOURS' represents the employee name.
    3. Identify which column in 'PLAN' represents the employee name.
    Return ONLY a Python dictionary: {{"income": 0.0, "hours_col": "", "plan_col": ""}}
    Data: {data_preview}
    """
    
    try:
        intel = eval(model.generate_content(intel_prompt).text)
        detected_income = intel['income']
        
        # 4. ACTUAL TRUTH CALCULATION [cite: 30, 144]
        df_hours['key'] = df_hours[intel['hours_col']].apply(normalize)
        df_plan['key'] = df_plan[intel['plan_col']].apply(normalize)
        
        # Merge for margin analysis [cite: 159]
        merged = pd.merge(df_hours, df_plan, on='key', how='left')
        
        # Calculate Labor Burn (Hours * Cost Rate) [cite: 32, 145]
        # We assume 'Cost Rate' exists or take an average [cite: 161]
        cost_col = next((c for c in merged.columns if 'Cost' in c), None)
        total_labor_cost = (merged['Hours'] * merged[cost_col]).sum()
        actual_margin = detected_income - total_labor_cost

        # 5. DASHBOARD & INSIGHTS [cite: 178, 183]
        st.header(f"Account Margin Status: {(actual_margin/detected_income)*100:.1f}%")
        
        col1, col2 = st.columns(2)
        col1.metric("Dynamic Income", f"${detected_income:,.2f}")
        col2.metric("Labor Burn", f"${total_labor_cost:,.2f}")

        # Visualize Seniority Drift [cite: 36, 146]
        st.subheader("Labor Distribution by Task")
        st.plotly_chart(px.bar(merged, x='Task', y='Hours', color=intel['hours_col']))

        # 6. EXECUTIVE SUMMARY (Resolving the 'Manual' burden) [cite: 37, 183]
        if st.button("Generate Executive Summary"):
            summary_prompt = f"Income: {detected_income}, Burn: {total_labor_cost}, Margin: {actual_margin}. Provide 3 bullet points for Becca/Raquel on account health."
            st.success(model.generate_content(summary_prompt).text)

    except Exception as e:
        st.error(f"Logic Error: {e}. Please ensure file headers are clear.")
