import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import io

# 1. AUTHENTICATION & PERSONA
# Setup for the automated executive summary logic discussed with Raquel
# Replace with your actual Gemini API Key from Google AI Studio
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-pro')

st.set_page_config(page_title="DEPT Financial Margin Engine", layout="wide")
st.title("📊 DEPT Financial Margin Engine")
st.markdown("Automated Margin Reporting & Seniority Drift Audit")

# 2. DATA CLEANING FUNCTIONS (Resolving the "Name Match" issue)
def normalize(name):
    if pd.isna(name): return ""
    return str(name).lower().strip().replace(" ", "").replace("'", "")

# 3. SIDEBAR: DATA INGESTION
with st.sidebar:
    st.header("1. Upload Raw Sources")
    # Raw Export from Harvest/DabApps as requested [cite: 43, 44]
    hours_file = st.file_uploader("Upload DabApps/Harvest Hours", type=['csv'])
    # The CPT/Pricing Plan with staff cost rates [cite: 32, 42]
    pricing_file = st.file_uploader("Upload Pricing Plan (CPT)", type=['csv'])
    
    st.header("2. Financial Constants")
    monthly_income = st.number_input("Monthly Recognized Income ($)", value=95000)

if hours_file and pricing_file:
    # Load and Normalize
    df_hours = pd.read_csv(hours_file)
    df_plan = pd.read_csv(pricing_file)
    
    # Cleaning columns based on the Fullscript/Plaid files
    df_hours['match_name'] = df_hours['Person'].apply(normalize)
    df_plan['match_name'] = df_plan['Name'].apply(normalize)
    
    # 4. THE MERGE: Mapping Cost Rates to Hours [cite: 14, 34]
    # This creates the "Actual Truth" margin report Raquel wants [cite: 30]
    df = pd.merge(df_hours, df_plan[['match_name', 'Cost Rate', 'Role', 'Level']], on='match_name', how='left')
    
    # Fallback logic for "Ghost Team" replacements not in CPT [cite: 34]
    df['Cost Rate'] = df['Cost Rate'].fillna(100.0) 
    df['Burned_Value'] = df['Hours'] * df['Cost Rate']
    
    # 5. DASHBOARD: MARGIN & DRIFT
    total_burn = df['Burned_Value'].sum()
    actual_margin = monthly_income - total_burn
    margin_pct = (actual_margin / monthly_income) * 100
    
    st.header("Account Financial Health")
    c1, c2, c3 = st.columns(3)
    c1.metric("Burned Labor Value", f"${total_burn:,.2f}")
    c2.metric("Actual Margin ($)", f"${actual_margin:,.2f}")
    c3.metric("Margin Efficiency", f"{margin_pct:.1f}%")

    # 6. RESOLUTION: Seniority Drift Audit [cite: 36]
    st.subheader("⚠️ Seniority Drift Audit")
    # Logic to flag high-cost staff doing administrative tasks
    drift = df[(df['Cost Rate'] > 150) & (df['Task'].str.contains('Ad Ops|Reporting', na=False, case=False))]
    if not drift.empty:
        st.warning(f"Detected {len(drift)} instances of Seniority Drift.")
        st.dataframe(drift[['Person', 'Task', 'Hours', 'Burned_Value']])

    # 7. AUTOMATED INSIGHTS: Executive Summary [cite: 13, 37]
    st.header("📝 AI-Generated Executive Summary")
    if st.button("Analyze & Generate Summary"):
        # Feed facts to Gemini
        facts = f"Margin is {margin_pct:.1f}%. Total Burn is ${total_burn}. Drift instances: {len(drift)}."
        prompt = f"Act as a DEPT Delivery Lead. Analyze these financials for Becca: {facts}. Provide a 3-bullet summary focused on resolution."
        
        response = model.generate_content(prompt)
        st.info(response.text)

    # 8. VISUALS
    st.subheader("Burn Distribution by Task")
    fig = px.pie(df, values='Burned_Value', names='Task', hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload the DabApps and CPT files to generate the financial report.")
