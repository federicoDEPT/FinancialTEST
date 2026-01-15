import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import plotly.express as px

# 1. SETUP: Use the NEW API
api_key = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=api_key)

st.title("📊 DEPTapps Strategic Margin Engine")
st.markdown("Automated 'Actual Truth' Reporting for Plaid & Fullscript")

# 2. UPLOAD: Excel & CSV support
with st.sidebar:
    st.header("Upload Files")
    h_file = st.file_uploader("DEPTapps/Harvest (Hours)", type=['xlsx', 'csv'])
    p_file = st.file_uploader("Pricing Plan (CPT)", type=['xlsx', 'csv'])
    b_file = st.file_uploader("Billing/Media Worksheet", type=['xlsx', 'csv'])

def load_any(file):
    if file.name.endswith('.csv'): 
        return pd.read_csv(file)
    return pd.read_excel(file)

if h_file and p_file and b_file:
    # Load the data
    df_h = load_any(h_file)
    df_p = load_any(p_file)
    df_b = load_any(b_file)

    # 3. THE "SMART" LAYER: Asking Gemini to map the data
    st.info("🤖 Gemini is interpreting your file structures and varied income...")
    
    snippet = f"""
    HOURS HEADERS: {list(df_h.columns)}
    PRICING HEADERS: {list(df_p.columns)}
    BILLING DATA SNIPPET: {df_b.head(15).to_string()}
    """

    prompt = f"""
    Act as a Financial Analyst. Look at these DEPT files:
    1. Which column in 'HOURS' contains the employee names?
    2. Which column in 'PRICING' contains the employee names?
    3. Which column in 'PRICING' contains the Cost Rate?
    4. What is the total 'Dept Fee' or 'Recognized Income' in the Billing snippet?
    
    Return ONLY a Python dictionary like this:
    {{"h_name": "col_name", "p_name": "col_name", "cost_col": "col_name", "income": 0.0}}
    
    Data: {snippet}
    """

    try:
        # NEW API call
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',  # Use the latest model
            contents=prompt
        )
        
        # Extract and clean the response
        intel = eval(response.text.strip().replace("```json", "").replace("```", ""))
        
        # 4. RUN THE MATH
        df_h['key'] = df_h[intel['h_name']].str.lower().str.strip()
        df_p['key'] = df_p[intel['p_name']].str.lower().str.strip()
        
        merged = pd.merge(df_h, df_p[['key', intel['cost_col']]], on='key', how='left')
        
        # Calculate Margin
        total_labor_cost = (merged['Hours'] * merged[intel['cost_col']]).sum()
        detected_income = intel['income']
        actual_margin = detected_income - total_labor_cost

        # 5. DASHBOARD
        c1, c2, c3 = st.columns(3)
        c1.metric("Dynamic Income", f"${detected_income:,.2f}")
        c2.metric("Labor Burn", f"${total_labor_cost:,.2f}")
        c3.metric("Actual Margin", f"{(actual_margin/detected_income)*100:.1f}%", 
                  delta=f"${actual_margin:,.2f}")

        # Task Distribution
        st.subheader("Labor Distribution by Task")
        task_hours = merged.groupby('Task')['Hours'].sum().reset_index()
        st.plotly_chart(px.bar(task_hours, x='Task', y='Hours'))

        # 6. EXECUTIVE SUMMARY
        if st.button("Generate Executive Summary"):
            summary_prompt = f"Income: {detected_income}, Cost: {total_labor_cost}, Margin: {actual_margin}. Provide 3 bullets for Becca."
            sum_resp = client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=summary_prompt
            )
            st.success(sum_resp.text)

    except Exception as e:
        st.error(f"Could not interpret files. Error: {e}")
