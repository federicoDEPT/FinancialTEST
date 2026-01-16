import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import json
import io

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="DEPT Financial Architect", layout="wide", page_icon="🧠")

# CSS for Professional "Deck" Look
st.markdown("""
<style>
    .metric-card { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; }
    h1 { font-family: 'Helvetica', sans-serif; letter-spacing: -1px; font-weight: 800; }
    .stFileUploader { padding: 20px; border: 2px dashed #cbd5e1; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Setup Gemini
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    # Force JSON output for machine-readable logic
    model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
except:
    st.error("⚠️ GOOGLE_API_KEY missing in Streamlit Secrets.")

# --- 2. INTELLIGENT DISCOVERY ENGINE ---

def read_file_preview(file):
    """Reads the first 30 rows of ANY file to let AI inspect the structure."""
    try:
        # Read as generic csv/excel first without assuming headers
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, header=None, nrows=30)
        else:
            df = pd.read_excel(file, header=None, nrows=30)
        return df.to_string()
    except:
        return ""

def identify_files_with_ai(file_previews):
    """
    Sends file snippets to Gemini. 
    Asks: 'Which file is the Billing Worksheet? Which is the Hours Log? Where do headers start?'
    """
    prompt = f"""
    You are a Forensic Data Analyst. I have uploaded multiple agency financial files. 
    Your job is to identify which file serves which purpose and find the header row index.

    FILE PREVIEWS:
    {json.dumps(file_previews)}

    TASKS:
    1. Find the **HOURS_LOG**: Look for 'Hours', 'Person', 'Date', 'Task', 'Notes'.
    2. Find the **PRICING_PLAN**: Look for 'Role', 'Rate', 'Cost', 'C Level', 'Name'.
    3. Find the **BILLING_SHEET**: Look for 'Recognized Income', 'Dept Fee', 'Total Fee', 'Media Spend'.
    
    RETURN JSON ONLY:
    {{
        "hours_file_name": "filename",
        "hours_header_row": 0,
        "pricing_file_name": "filename",
        "pricing_header_row": 0,
        "billing_file_name": "filename",
        "billing_header_row": 0
    }}
    If a file is missing, return null for that field.
    """
    try:
        resp = model.generate_content(prompt)
        return json.loads(resp.text)
    except Exception as e:
        st.error(f"AI Classification Error: {e}")
        return None

def extract_data_smart(file, header_row, file_type):
    """Re-reads the file using the AI-detected header row."""
    try:
        file.seek(0) # Reset pointer
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, header=header_row)
        else:
            df = pd.read_excel(file, header=header_row)
        return df
    except:
        return None

def get_column_mapping(df_h, df_p, df_b):
    """Asks AI to map the specific column names for joining."""
    snippet_h = df_h.head(3).to_string()
    snippet_p = df_p.head(3).to_string()
    snippet_b = df_b.head(10).to_string()

    prompt = f"""
    Map these columns for a financial report.
    
    HOURS LOG: {snippet_h}
    PRICING: {snippet_p}
    BILLING: {snippet_b}

    Identify:
    1. 'h_name': The Person/Employee column in Hours.
    2. 'p_name': The Person/Employee column in Pricing.
    3. 'cost_col': The Cost Rate column in Pricing (Look for 'Cost', 'Rate', 'Hourly').
    4. 'task_col': The Task/Client column in Hours.
    5. 'total_revenue': The TOTAL Recognized Revenue/Dept Fee for the current month from Billing (Sum relevant rows if needed). Return a FLOAT.

    JSON ONLY:
    {{ "h_name": "", "p_name": "", "cost_col": "", "task_col": "", "total_revenue": 0.0 }}
    """
    resp = model.generate_content(prompt)
    return json.loads(resp.text)

# --- 3. MAIN INTERFACE ---

st.title("📊 DEPT Strategic Financial Architect")
st.markdown("### The Actual Truth Engine")
st.markdown("Upload **ALL** your account files (Billing, Pricing, Harvest logs). The AI will sort them out.")

# 1. UNIVERSAL UPLOAD ZONE
uploaded_files = st.file_uploader("Drop all files here (Excel/CSV)", accept_multiple_files=True)

if uploaded_files:
    # A. PREVIEW PHASE
    file_map = {f.name: f for f in uploaded_files}
    previews = {name: read_file_preview(f) for name, f in file_map.items()}

    if st.button("Analyze & Reconcile Files") or True: # Auto-run
        with st.spinner("🧠 AI is analyzing file types and structures..."):
            # Step 1: Identify which file is which
            structure = identify_files_with_ai(previews)
        
        if structure:
            # Step 2: Extract Data using intelligent headers
            try:
                # Load Hours
                f_h = file_map.get(structure['hours_file_name'])
                df_h = extract_data_smart(f_h, structure['hours_header_row'], "HOURS")

                # Load Pricing
                f_p = file_map.get(structure['pricing_file_name'])
                df_p = extract_data_smart(f_p, structure['pricing_header_row'], "PRICING")

                # Load Billing
                f_b = file_map.get(structure['billing_file_name'])
                df_b = extract_data_smart(f_b, structure['billing_header_row'], "BILLING")

                st.success(f"✅ Identified: **Hours** ({f_h.name}), **Pricing** ({f_p.name}), **Billing** ({f_b.name})")

                # Step 3: Map Columns & Calc
                with st.spinner("💰 AI is extracting revenue and mapping cost rates..."):
                    mapping = get_column_mapping(df_h, df_p, df_b)

                # --- B. THE FINANCIAL ENGINE ---
                
                # 1. Normalize & Merge
                h_col = mapping['h_name']
                p_col = mapping['p_name']
                cost_col = mapping['cost_col']
                
                df_h['clean_name'] = df_h[h_col].astype(str).str.lower().str.strip()
                df_p['clean_name'] = df_p[p_col].astype(str).str.lower().str.strip()
                
                merged = pd.merge(df_h, df_p, on='clean_name', how='left')
                
                # 2. Burn Calculation
                # Clean cost rate (remove '$', handle errors)
                merged[cost_col] = pd.to_numeric(merged[cost_col].astype(str).str.replace('$', '', regex=False), errors='coerce').fillna(0)
                merged['Burn'] = merged['Hours'] * merged[cost_col]

                # 3. Filter Logic (Exclude SEO/Data Science)
                task_col = mapping['task_col']
                active_work = merged[~merged[task_col].astype(str).str.contains("SEO|Data Science", case=False, na=False)]

                # 4. Final Totals
                total_rev = mapping['total_revenue']
                total_burn = active_work['Burn'].sum()
                margin = total_rev - total_burn
                margin_pct = (margin / total_rev) * 100 if total_rev else 0

                # --- C. DASHBOARD VISUALS ---
                
                st.markdown("---")
                # KPI ROW
                k1, k2, k3 = st.columns(3)
                k1.metric("Recognized Revenue", f"${total_rev:,.0f}")
                k2.metric("Total Labor Burn", f"${total_burn:,.0f}", delta=f"{(total_burn/total_rev)*100:.1f}% Burn Rate", delta_color="inverse")
                k3.metric("Net Margin %", f"{margin_pct:.1f}%", delta=f"{margin_pct-60:.1f}% vs Target (60%)")

                # CHART ROW
                c_chart, c_audit = st.columns([2, 1])
                
                with c_chart:
                    st.subheader("Profitability Gap")
                    chart_df = pd.DataFrame({
                        "Type": ["Revenue", "Labor Burn"],
                        "Amount": [total_rev, total_burn],
                        "Color": ["#6366f1", "#f43f5e"]
                    })
                    fig = px.bar(chart_df, x="Type", y="Amount", color="Type", 
                                 text_auto="$.2s", title="Revenue vs. Cost Reality",
                                 color_discrete_map={"Revenue": "#6366f1", "Labor Burn": "#f43f5e"})
                    st.plotly_chart(fig, use_container_width=True)

                with c_audit:
                    st.subheader("Strategic Audit")
                    # GHOST TEAM (0 Cost Rate)
                    ghosts = merged[(merged[cost_col] == 0) & (merged['Hours'] > 0)]
                    if not ghosts.empty:
                        st.error(f"👻 {len(ghosts['clean_name'].unique())} Ghost Members Detected")
                        st.dataframe(ghosts.groupby(h_col)['Hours'].sum().reset_index().sort_values('Hours', ascending=False), height=200)
                    else:
                        st.success("No Ghost Team detected.")

                # SENIORITY DRIFT
                st.subheader("Seniority Drift Analysis")
                seniors = merged[merged[cost_col] >= 150]
                drift = seniors[seniors[task_col].astype(str).str.contains("Admin|Report|Internal", case=False, na=False)]
                
                if not drift.empty:
                    st.warning("⚠️ High-Cost Staff performing Low-Value Tasks")
                    st.dataframe(drift.groupby([h_col, task_col])['Hours'].sum().reset_index())

            except Exception as e:
                st.error(f"Processing Error: {e}. AI successfully identified files but failed to parse data rows.")
        else:
            st.warning("Could not identify files. Please ensure you uploaded Hours, Pricing, and Billing files.")
