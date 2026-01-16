import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import json

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="DEPT Financial Architect", layout="wide", page_icon="📊")

# Custom CSS to match the professional "Dashboard" look
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: transparent !important;
    }
    /* Header Styling */
    h1 { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; letter-spacing: -1px; }
    h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 600; color: #475569; }
</style>
""", unsafe_allow_html=True)

# --- 2. SETUP GEMINI (THE BRAIN) ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception:
    st.error("⚠️ GOOGLE_API_KEY missing in Streamlit Secrets. AI features will be disabled.")

# --- 3. HELPER FUNCTIONS ---
def normalize(name):
    """Normalize names to ensure 'Mike' matches 'Michael' logic could be added here."""
    if pd.isna(name): return ""
    return str(name).lower().strip().replace(" ", "").replace("'", "")

@st.cache_data
def load_file(file):
    if file.name.endswith('.csv'): return pd.read_csv(file)
    return pd.read_excel(file)

def get_ai_mapping(df_h, df_p, df_b):
    """Asks Gemini to map columns and find variable income to avoid hardcoding crashes."""
    prompt = f"""
    Act as a Data Engineer. Analyze these 3 file snippets:
    1. HOURS: {list(df_h.columns)}
    2. PRICING: {list(df_p.columns)}
    3. BILLING DATA: {df_b.head(10).to_string()}

    Task:
    1. Identify the 'Person' name column in HOURS.
    2. Identify the 'Person' name column in PRICING.
    3. Identify the 'Cost Rate' column in PRICING.
    4. Identify the 'Task' or 'Role' column in HOURS.
    5. Find the TOTAL 'Recognized Income' or 'Dept Fee' for this month from BILLING (sum if multiple rows).

    Return JSON ONLY:
    {{
        "h_name": "col_name", 
        "p_name": "col_name", 
        "cost_col": "col_name", 
        "task_col": "col_name",
        "total_income": 0.0
    }}
    """
    try:
        resp = model.generate_content(prompt)
        return json.loads(resp.text.replace("```json", "").replace("```", "").strip())
    except:
        return None

# --- 4. MAIN APPLICATION ---

st.title("📊 DEPT Strategic Financial Architect")
st.markdown("**Reconciliation Engine:** Labor Burn vs. Variable Media Revenue")

# SIDEBAR: The Control Center
with st.sidebar:
    st.header("1. Upload Source Data")
    h_file = st.file_uploader("DEPTapps / Harvest (Hours)", type=['xlsx', 'csv'])
    p_file = st.file_uploader("CPT / Pricing Plan (Rates)", type=['xlsx', 'csv'])
    b_file = st.file_uploader("Billing Worksheet (Income)", type=['xlsx', 'csv'])
    
    st.divider()
    st.header("2. Strategic Filters")
    # Default filters based on your report requirements
    excluded_tasks = st.multiselect(
        "Exclude Workstreams", 
        ['SEO', 'Data Science', 'Strategy', 'Paid Social'],
        default=['SEO', 'Data Science']
    )

if h_file and p_file and b_file:
    # A. LOAD DATA
    df_h = load_file(h_file)
    df_p = load_file(p_file)
    df_b = load_file(b_file)

    # B. AI MAPPING (The "Smart" Layer)
    with st.spinner("🤖 AI is analyzing file structures and reconciling income..."):
        mapping = get_ai_mapping(df_h, df_p, df_b)
    
    if mapping:
        # C. DATA PROCESSING
        # 1. Normalize Names for Merging
        df_h['join_key'] = df_h[mapping['h_name']].apply(normalize)
        df_p['join_key'] = df_p[mapping['p_name']].apply(normalize)
        
        # 2. Merge Hours with Cost Rates
        merged = pd.merge(df_h, df_p, on='join_key', how='left')
        
        # 3. Calculate Burn
        # Fill missing rates with 0 to detect Ghosts later
        cost_col = mapping['cost_col']
        merged[cost_col] = pd.to_numeric(merged[cost_col], errors='coerce').fillna(0)
        merged['Burn'] = merged['Hours'] * merged[cost_col]
        
        # 4. Apply Filters (The "Logic Gate")
        # We filter out rows where the Task column contains the excluded keywords
        task_col = mapping['task_col']
        # Safe filter: only keep rows where Task does NOT contain excluded terms
        pattern = '|'.join(excluded_tasks) if excluded_tasks else "ZZZZZ" # ZZZZZ matches nothing
        filtered_df = merged[~merged[task_col].astype(str).str.contains(pattern, case=False, na=False)]

        # 5. Financial Totals
        total_income = mapping['total_income']
        total_burn = filtered_df['Burn'].sum()
        margin_dollars = total_income - total_burn
        margin_pct = (margin_dollars / total_income) * 100
        target_margin = 60.0

        # --- D. DASHBOARD UI (REPLICATING YOUR PPT) ---

        # ZONE 1: EXECUTIVE PULSE
        st.subheader("1. Executive Pulse")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.metric("Recognized Income (Billing)", f"${total_income:,.0f}", delta="From Billing Worksheet")
        with c2:
            st.metric("Total Labor Burn", f"${total_burn:,.0f}", delta=f"{total_burn/total_income*100:.1f}% of Rev", delta_color="inverse")
        with c3:
            st.metric("Net Margin %", f"{margin_pct:.1f}%", delta=f"{margin_pct - target_margin:.1f}% vs Target")

        # ZONE 2: THE PROFITABILITY GAP (Chart)
        st.divider()
        c_chart, c_table = st.columns([2, 1])
        
        with c_chart:
            st.subheader("2. The Profitability Gap")
            # Creating a clean bar chart like your React mockup
            chart_data = pd.DataFrame({
                "Metric": ["Revenue", "Labor Burn"],
                "Amount": [total_income, total_burn],
                "Color": ["#6366f1", "#f43f5e"] # Indigo and Rose
            })
            fig = px.bar(chart_data, x="Metric", y="Amount", color="Metric", 
                         color_discrete_map={"Revenue": "#6366f1", "Labor Burn": "#f43f5e"},
                         text_auto='.2s', title="Revenue vs. Cost Reality")
            fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with c_table:
            st.subheader("Workstream Breakdown")
            # Group by Task/Role to show where the money is going
            ws_data = filtered_df.groupby(task_col)[['Hours', 'Burn']].sum().sort_values('Burn', ascending=False).head(5)
            st.dataframe(ws_data.style.format({"Burn": "${:,.0f}", "Hours": "{:,.1f}"}), use_container_width=True)

        # ZONE 3: THE AUDIT (GHOSTS & DRIFT)
        st.divider()
        st.subheader("3. Strategic Audit & Anomaly Detection")
        
        audit_c1, audit_c2 = st.columns(2)
        
        with audit_c1:
            st.markdown("### 👻 Ghost Team Detector")
            st.caption("Staff billing time but MISSING from CPT Cost Table")
            # Ghosts are people with 0 Cost Rate (failed merge) but > 0 Hours
            ghosts = merged[(merged[cost_col] == 0) & (merged['Hours'] > 0)]
            if not ghosts.empty:
                st.dataframe(ghosts[[mapping['h_name'], 'Hours', task_col]].groupby(mapping['h_name']).sum())
            else:
                st.success("No Ghost Team members detected.")

        with audit_c2:
            st.markdown("### 📉 Seniority Drift")
            st.caption("High-Cost Staff (>$150/hr) doing Admin/Reporting")
            # Logic: Cost > 150 AND Task contains 'Admin' or 'Report'
            high_cost = filtered_df[filtered_df[cost_col] > 150]
            drift = high_cost[high_cost[task_col].str.contains("Admin|Report|Internal|Meeting", case=False, na=False)]
            
            if not drift.empty:
                drift_summary = drift.groupby(mapping['h_name'])[['Hours', 'Burn']].sum().sort_values('Burn', ascending=False)
                st.dataframe(drift_summary.style.format({"Burn": "${:,.0f}"}))
            else:
                st.success("No significant seniority drift detected.")

        # ZONE 4: AI EXECUTIVE SUMMARY
        st.divider()
        if st.button("Generate Senior Executive Summary"):
            with st.spinner("Analyzing financial narrative..."):
                summary_prompt = f"""
                You are a Financial Controller. Write a 3-bullet executive summary based on this data:
                - Income: ${total_income}
                - Burn: ${total_burn}
                - Margin: {margin_pct}% (Target: 60%)
                - Top Burn Driver: {ws_data.index[0]} (${ws_data.iloc[0]['Burn']})
                
                Focus on the 'Turnaround Story' or 'Risk Factors'. Be concise and professional.
                """
                summary = model.generate_content(summary_prompt).text
                st.info(summary)
            
    else:
        st.error("Could not interpret file structure. Please ensure columns have clear headers like 'Person', 'Hours', 'Rate'.")

else:
    # LANDING PAGE STATE
    st.info("👋 Welcome to the DEPT Financial Engine. Upload your monthly files to begin the strategic audit.")
