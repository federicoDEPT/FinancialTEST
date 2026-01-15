import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai

# FIX 1: Use the updated model name
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash') # Updated from gemini-pro

st.title("📊 DEPT Financial Margin Engine")

# Normalization function for name matching [cite: 33, 34]
def normalize(name):
    if pd.isna(name): return ""
    return str(name).lower().strip().replace(" ", "").replace("'", "")

with st.sidebar:
    st.header("1. Data Ingestion")
    hours_file = st.file_uploader("Upload Raw DEPTapps Export", type=['xlsx', 'xls', 'csv'])
    pricing_file = st.file_uploader("Upload Pricing Plan (CPT)", type=['xlsx', 'xls', 'csv'])

if hours_file and pricing_file:
    df_hours = pd.read_excel(hours_file) if not hours_file.name.endswith('.csv') else pd.read_csv(hours_file)
    df_plan = pd.read_excel(pricing_file) if not pricing_file.name.endswith('.csv') else pd.read_csv(pricing_file)

    # FIX 2: Flexible column detection for 'Person' or 'Employee'
    hours_col = next((c for c in df_hours.columns if c in ['Person', 'Employee', 'User']), None)
    plan_col = next((c for c in df_plan.columns if c in ['Name', 'Employee', 'Full Name']), None)

    if hours_col and plan_col:
        df_hours['match_name'] = df_hours[hours_col].apply(normalize)
        df_plan['match_name'] = df_plan[plan_col].apply(normalize)

        # 4. THE MERGE: Mapping Cost Rates to Hours [cite: 12, 14, 34]
        df = pd.merge(df_hours, df_plan, on='match_name', how='left')
        
        # Financial Calculations [cite: 12, 30]
        st.success("Data merged successfully!")
        st.dataframe(df.head()) # Show a preview
    else:
        st.error(f"Could not find matching name columns. Found: {list(df_hours.columns)}")
