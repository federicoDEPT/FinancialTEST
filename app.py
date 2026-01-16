import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
from google.genai import types
import json

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="DEPT Financial Architect", layout="wide", page_icon="🧠")

st.markdown("""
<style>
    .metric-card { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; }
    h1 { font-family: 'Helvetica', sans-serif; letter-spacing: -1px; font-weight: 800; }
    .stFileUploader { padding: 20px; border: 2px dashed #cbd5e1; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 2. SETUP (NEW SDK) ---
try:
    # Initialize the modern Client (v2)
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"API Key Error: {e}")

# --- 3. INTELLIGENT FUNCTIONS ---

def read_file_preview(file):
    """Reads first 30 rows to let AI verify file content."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, header=None, nrows=30)
        else:
            df = pd.read_excel(file, header=None, nrows=30)
        return df.to_string()
    except:
        return ""

def identify_files_with_ai(file_previews):
    """Asks AI to identify which file is which based on content."""
    prompt = f"""
    You are a Forensic Data Analyst. Identify the purpose of these files based on their content previews.

    FILES:
    {json.dumps(file_previews)}

    TASKS:
    1. Find 'hours_file': Looks for 'Person', 'Hours', 'Task', 'Date'.
    2. Find 'pricing_file': Looks for 'Role', 'Rate', 'Cost', 'C Level'.
    3. Find 'billing_file': Looks for 'Recognized Income', 'Dept Fee', 'Media Spend'.

    RETURN JSON ONLY:
    {{
        "hours_file": "filename",
        "pricing_file": "filename",
        "billing_file": "filename",
        "header_row_hours": 0,
        "header_row_pricing": 0,
        "header_row_billing": 0
    }}
    """
    try:
        # NEW SDK SYNTAX: client.models.generate_content
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        st.error(f"AI Classification Error: {e}")
        return None

def extract_data(file, header_row):
    """Reloads the file using the correct header row detected by AI."""
    try:
        file.seek(0)
        if file.name.endswith('.csv'):
            return pd.read_csv(file, header=header_row)
        return pd.read_excel(file, header=header_row)
    except:
        return None

def get_mapping(df_h, df_p, df_b):
    """Maps columns and finds total revenue."""
    snippet_h = df_h.head(3).to_string()
    snippet_p = df_p.head(3).to_string()
    snippet_b = df_b.head(10).to_string()

    prompt = f"""
    Map the columns for these 3 datasets:
    HOURS: {snippet_h}
    PRICING: {snippet_p}
    BILLING: {snippet_b}

    Find:
    1. 'h_col': Name column in Hours.
    2. 'p_col': Name column in Pricing.
    3. 'cost_col': Cost Rate column in Pricing.
    4. 'task_col': Task column in Hours.
    5. 'total_revenue': Total Recognized Revenue/Dept Fee (float).

    RETURN JSON: {{ "h_col": "", "p_col": "", "cost_col": "", "task_col": "", "total_revenue": 0.0 }}
    """
    
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return json.loads(response.text)

# --- 4. MAIN DASHBOARD ---

st.title("📊 DEPT Strategic Financial Architect")
st.markdown("### The Actual Truth Engine")

uploaded_files = st.file_uploader("Drop all files (Hours, Pricing, Billing)", accept_multiple_files=True)

if uploaded_files:
    file_map = {f.name: f for f in uploaded_files}
    previews = {name: read_file_preview(f) for name, f in file_map.items()}

    if st.button("Analyze Files") or True:
        with st.spinner("🧠 AI is identifying file structures..."):
            structure = identify_files_with_ai(previews)
        
        if structure and structure['hours_file']:
            # Load Data
            df_h = extract_data(file_map[structure['hours_file']], structure['header_row_hours'])
            df_p = extract_data(file_map[structure['pricing_file']], structure['header_row_pricing'])
            df_b = extract_data(file_map[structure['billing_file']], structure['header_row_billing'])

            if df_h is not None and df_p is not None:
                with st.spinner("💰 Mapping financials..."):
                    mapping = get_mapping(df_h, df_p, df_b)

                # Processing
                h_col, p_col, cost_col = mapping['h_col'], mapping['p_col'], mapping['cost_col']
                
                # Normalize & Merge
                df_h['key'] = df_h[h_col].astype(str).str.lower().str.strip()
                df_p['key'] = df_p[p_col].astype(str).str.lower().str.strip()
                merged = pd.merge(df_h, df_p, on='key', how='left')
                
                # Burn Calc
                merged[cost_col] = pd.to_numeric(merged[cost_col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
                merged['Burn'] = merged['Hours'] * merged[cost_col]

                # Financials
                revenue = mapping['total_revenue']
                burn = merged['Burn'].sum()
                margin = revenue - burn
                margin_pct = (margin / revenue) * 100 if revenue else 0

                # --- VISUALS ---
                st.markdown("---")
                k1, k2, k3 = st.columns(3)
                k1.metric("Recognized Revenue", f"${revenue:,.0f}")
                k2.metric("Labor Burn", f"${burn:,.0f}", delta=f"{(burn/revenue)*100:.1f}% Burn", delta_color="inverse")
                k3.metric("Net Margin", f"{margin_pct:.1f}%", delta=f"{margin_pct-60:.1f}% vs Target")

                # Gap Chart
                st.subheader("Profitability Gap")
                chart_df = pd.DataFrame({
                    "Type": ["Revenue", "Labor Burn"],
                    "Amount": [revenue, burn],
                    "Color": ["#6366f1", "#f43f5e"]
                })
                fig = px.bar(chart_df, x="Amount", y="Type", orientation='h', text_auto="$.2s", color="Type", color_discrete_map={"Revenue": "#6366f1", "Labor Burn": "#f43f5e"})
                st.plotly_chart(fig, use_container_width=True)

                # Ghost Audit
                st.subheader("👻 Ghost Team Audit (Billing but not in Pricing)")
                ghosts = merged[(merged[cost_col] == 0) & (merged['Hours'] > 0)]
                if not ghosts.empty:
                    st.dataframe(ghosts.groupby(h_col)['Hours'].sum().reset_index().sort_values('Hours', ascending=False))
                else:
                    st.success("No Ghosts detected.")
        else:
            st.warning("Could not identify all 3 required files. Please ensure you uploaded Hours, Pricing, and Billing.")
