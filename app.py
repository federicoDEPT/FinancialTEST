import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai

# Setup Gemini for Dynamic Interpretation
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

st.set_page_config(page_title="DEPT Financial Engine", layout="wide")
st.title("📊 DEPT Financial Margin Engine")

with st.sidebar:
    st.header("1. Data Ingestion")
    # Supports .xlsx and .xls as requested
    hours_file = st.file_uploader("Upload Raw DEPTapps Export", type=['xlsx', 'xls', 'csv'])
    pricing_file = st.file_uploader("Upload Pricing Plan (CPT)", type=['xlsx', 'xls', 'csv'])
    # Upload the Billing Worksheet where income varies month-to-month
    income_file = st.file_uploader("Upload Billing Worksheet / Media Plan", type=['xlsx', 'xls', 'csv'])

if hours_file and pricing_file and income_file:
    # 2. LOAD DATA (Excel Compatible)
    df_hours = pd.read_excel(hours_file) if not hours_file.name.endswith('.csv') else pd.read_csv(hours_file)
    df_plan = pd.read_excel(pricing_file) if not pricing_file.name.endswith('.csv') else pd.read_csv(pricing_file)
    df_income = pd.read_excel(income_file) if not income_file.name.endswith('.csv') else pd.read_csv(income_file)

    # 3. DYNAMIC INCOME INTERPRETATION
    # We send a snippet of the income file to Gemini to find the exact revenue for the period
    income_sample = df_income.head(50).to_string()
    income_prompt = f"""
    Look at this financial data. Find the 'Recognized Income', 'Dept Fee', or 'Commission' 
    that varies based on media spend or SOW. 
    Identify the total for the current reporting month. 
    Return ONLY the numerical value: {income_sample}
    """
    
    # AI extracts the variable income
    detected_income = float(model.generate_content(income_prompt).text.replace('$', '').replace(',', ''))
    
    # 4. MARGIN CALCULATION (The "Actual Truth")
    # Normalizing names to ensure the merge works mechanically
    df_hours['match_name'] = df_hours['Person'].str.lower().str.strip()
    df_plan['match_name'] = df_plan['Name'].str.lower().str.strip()
    
    df_merged = pd.merge(df_hours, df_plan[['match_name', 'Cost Rate']], on='match_name', how='left')
    total_labor_cost = (df_merged['Hours'] * df_merged['Cost Rate']).sum()
    actual_margin = detected_income - total_labor_cost

    # 5. DASHBOARD: RESOLVING THE MARGIN GAP
    st.header(f"Financial Reconciliation: ${detected_income:,.2f} Revenue")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Dynamic Income", f"${detected_income:,.2f}")
    col2.metric("Total Labor Cost", f"${total_labor_cost:,.2f}")
    col3.metric("Margin Efficiency", f"{(actual_margin/detected_income)*100:.1f}%")

    # 6. EXECUTIVE SUMMARY (Automated Insights)
    if st.button("Generate Executive Summary"):
        # We pass the variable data to Gemini to write the 'Why'
        summary_prompt = f"""
        Income: ${detected_income}. Labor Cost: ${total_labor_cost}. Margin: ${actual_margin}.
        Provide 3 bullet points for Becca and Raquel. 
        Focus on whether the variable media-spend revenue is covering the labor intensity 
        and if 'Seniority Drift' is impacting the margin.
        """
        st.info(model.generate_content(summary_prompt).text)
