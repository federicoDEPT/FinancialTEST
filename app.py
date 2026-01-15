import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import plotly.express as px

# 1. SETUP: Configure the NEW API
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
    """Load CSV or Excel files"""
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
    
    # Create a snippet of your data for analysis
    snippet = f"""
    HOURS FILE HEADERS: {list(df_h.columns)}
    PRICING FILE HEADERS: {list(df_p.columns)}
    BILLING DATA PREVIEW:
    {df_b.head(15).to_string()}
    """

    # Prompt for Gemini to analyze the structure
    prompt = f"""
    You are a Financial Data Analyst. Analyze these DEPT accounting files:
    
    1. Which column in 'HOURS' contains the employee names?
    2. Which column in 'PRICING' contains the employee names?
    3. Which column in 'PRICING' contains the hourly Cost Rate?
    4. What is the total 'Dept Fee' or 'Recognized Income' in the Billing data? (Sum all revenue)
    
    Return ONLY a valid Python dictionary in this exact format (no markdown, no explanation):
    {{"h_name": "column_name", "p_name": "column_name", "cost_col": "column_name", "income": 12345.67}}
    
    DATA TO ANALYZE:
    {snippet}
    """

    try:
        # NEW API CALL - Using the modern google-genai package
        response = client.models.generate_content(
            model='gemini-1.5-flash',  # Use a supported model
            contents=prompt
        )
        
        # Clean and parse the response
        response_text = response.text.strip()
        # Remove any markdown code blocks if present
        response_text = response_text.replace("```python", "").replace("```json", "").replace("```", "")
        
        # Parse the dictionary
        intel = eval(response_text)
        
        st.success("✅ File structure analyzed successfully!")
        
        # 4. RUN THE MATH: Using AI's intelligent mapping
        # Normalize names (handles "Michael" vs "Mike" issues)
        df_h['key'] = df_h[intel['h_name']].str.lower().str.strip()
        df_p['key'] = df_p[intel['p_name']].str.lower().str.strip()
        
        # Merge hours with cost rates
        merged = pd.merge(
            df_h, 
            df_p[['key', intel['cost_col']]], 
            on='key', 
            how='left'
        )
        
        # Calculate the ACTUAL MARGIN (The Truth)
        total_labor_cost = (merged['Hours'] * merged[intel['cost_col']]).sum()
        detected_income = float(intel['income'])
        actual_margin = detected_income - total_labor_cost
        margin_percent = (actual_margin / detected_income) * 100 if detected_income > 0 else 0

        # 5. DASHBOARD: Display Executive Metrics
        st.subheader("💰 Financial Overview")
        c1, c2, c3 = st.columns(3)
        
        c1.metric(
            "Client Revenue", 
            f"${detected_income:,.2f}",
            help="Total income from billing/media worksheet"
        )
        c2.metric(
            "Labor Cost", 
            f"${total_labor_cost:,.2f}",
            help="Total hours × cost rates"
        )
        c3.metric(
            "Actual Margin", 
            f"{margin_percent:.1f}%", 
            delta=f"${actual_margin:,.2f}",
            help="Net profit after labor costs"
        )

        # Show the merged data for verification
        with st.expander("🔍 View Detailed Labor Data"):
            st.dataframe(merged)

        # 6. VISUALIZATIONS
        st.subheader("📊 Labor Distribution Analysis")
        
        # Task Distribution (addresses Raquel's concerns)
        if 'Task' in merged.columns:
            task_summary = merged.groupby('Task')['Hours'].sum().reset_index()
            task_summary = task_summary.sort_values('Hours', ascending=False)
            
            fig = px.bar(
                task_summary, 
                x='Task', 
                y='Hours',
                title="Hours by Task Type",
                color='Hours',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Employee Distribution
        if intel['h_name'] in merged.columns:
            employee_summary = merged.groupby(intel['h_name'])['Hours'].sum().reset_index()
            employee_summary = employee_summary.sort_values('Hours', ascending=False)
            
            fig2 = px.bar(
                employee_summary,
                x=intel['h_name'],
                y='Hours',
                title="Hours by Team Member",
                color='Hours'
            )
            st.plotly_chart(fig2, use_container_width=True)

        # 7. EXECUTIVE SUMMARY (AI-Generated Insights)
        st.subheader("📝 Executive Summary")
        if st.button("🤖 Generate AI Summary for Becca"):
            summary_prompt = f"""
            Create a concise executive summary for Becca (leadership) with these metrics:
            - Total Revenue: ${detected_income:,.2f}
            - Labor Costs: ${total_labor_cost:,.2f}
            - Net Margin: ${actual_margin:,.2f} ({margin_percent:.1f}%)
            
            Provide exactly 3 bullet points:
            1. Financial health assessment
            2. Key insight about cost efficiency
            3. One actionable recommendation
            
            Keep it professional and data-driven.
            """
            
            summary_response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=summary_prompt
            )
            
            st.info(summary_response.text)

    except Exception as e:
        st.error(f"⚠️ Could not interpret files. Please check your data structure.")
        st.exception(e)
        
        # Helpful debugging info
        with st.expander("🔧 Debug Information"):
            st.write("**Hours File Columns:**", list(df_h.columns))
            st.write("**Pricing File Columns:**", list(df_p.columns))
            st.write("**Billing File Sample:**")
            st.dataframe(df_b.head())

else:
    st.info("👈 Please upload all three files to begin analysis")
