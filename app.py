import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import io

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="DEPT Financial Architect", layout="wide", page_icon="🧠")

st.markdown("""
<style>
    .metric-card { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; }
    h1 { font-family: 'Helvetica', sans-serif; letter-spacing: -1px; font-weight: 800; }
    .stFileUploader { padding: 20px; border: 2px dashed #cbd5e1; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 2. THE MANUAL API CONNECTION (UNBREAKABLE) ---

def ask_gemini(prompt):
    """
    Direct REST API connection. Bypasses library conflicts.
    """
    api_key = st.secrets["GOOGLE_API_KEY"]
    # We use the reliable 1.5 Flash endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "response_mime_type": "application/json"
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            st.error(f"API Error ({response.status_code}): {response.text}")
            return None
        
        # Parse the nested JSON response from Google
        result = response.json()
        text_content = result['candidates'][0]['content']['parts'][0]['text']
        return json.loads(text_content)
    except Exception as e:
        st.error(f"Connection Failed: {e}")
        return None

# --- 3. INTELLIGENT DISCOVERY ---

def read_file_preview(file):
    """Reads first 20 rows of ANY file to let AI inspect structure."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, header=None, nrows=20)
        else:
            df = pd.read_excel(file, header=None, nrows=20)
        return df.to_string()
    except:
        return ""

def identify_files_smartly(previews):
    """
    Sends file snippets to Gemini to figure out which is which.
    """
    prompt = f"""
    You are a Data Architect. I have uploaded multiple files. Identify which file is which based on content.

    FILE PREVIEWS:
    {json.dumps(previews)}

    TASKS:
    1. Find 'hours_file': Look for 'Person', 'Hours', 'Date', 'Task', 'Notes'.
    2. Find 'pricing_file': Look for 'Role', 'Rate', 'Cost', 'C Level', 'Name'.
    3. Find 'billing_file': Look for 'Recognized Income', 'Dept Fee', 'Media Spend', 'Total Fee'.
    
    RETURN JSON ONLY:
    {{
        "hours_file": "filename",
        "pricing_file": "filename",
        "billing_file": "filename",
        "h_header": 0,
        "p_header": 0,
        "b_header": 0
    }}
    If header seems to be on row 1, return 0 (0-index). If row 13, return 12.
    """
    return ask_gemini(prompt)

def extract_clean_data(file, header_row):
    """Reloads file with correct header."""
    try:
        file.seek(0)
        if file.name.endswith('.csv'):
            return pd.read_csv(file, header=header_row)
        return pd.read_excel(file, header=header_row)
    except:
        return None

def get_column_map(df_h, df_p, df_b):
    """Asks AI to map the columns for joining."""
    s_h = df_h.head(3).to_string()
    s_p = df_p.head(3).to_string()
    s_b = df_b.head(10).to_string()

    prompt = f"""
    Map the columns for these datasets.
    
    HOURS: {s_h}
    PRICING: {s_p}
    BILLING: {s_b}

    Identify EXACT column names:
    1. 'h_name': Employee Name in Hours.
    2. 'p_name': Employee Name in Pricing.
    3. 'cost_col': Cost Rate in Pricing.
    4. 'task_col': Task/Role in Hours.
    5. 'revenue': The TOTAL Recognized Revenue/Dept Fee (float) for this month.

    RETURN JSON: {{ "h_name": "", "p_name": "", "cost_col": "", "task_col": "", "revenue": 0.0 }}
    """
    return ask_gemini(prompt)

# --- 4. MAIN DASHBOARD ---

st.title("📊 DEPT Strategic Financial Architect")
st.markdown("### The Actual Truth Engine")
st.info("Upload ALL your files below. The AI will sort them out.")

uploaded_files = st.file_uploader("Drop Files Here", accept_multiple_files=True)

if uploaded_files:
    file_map = {f.name: f for f in uploaded_files}
    
    # 1. GENERATE PREVIEWS
    previews = {name: read_file_preview(f) for name, f in file_map.items()}

    if st.button("Run Auto-Reconciliation") or True:
        with st.spinner("🧠 AI is analyzing file types..."):
            structure = identify_files_smartly(previews)

        if structure and structure['hours_file']:
            st.success(f"✅ Identified: Hours ({structure['hours_file']}) | Pricing ({structure['pricing_file']}) | Billing ({structure['billing_file']})")
            
            # 2. LOAD DATA
            df_h = extract_clean_data(file_map[structure['hours_file']], structure['h_header'])
            df_p = extract_clean_data(file_map[structure['pricing_file']], structure['p_header'])
            df_b = extract_clean_data(file_map[structure['billing_file']], structure['b_header'])

            if df_h is not None and df_p is not None:
                # 3. MAP & CALCULATE
                with st.spinner("💰 Calculating Actual Margin..."):
                    mapping = get_column_map(df_h, df_p, df_b)
                
                if mapping:
                    # Normalize for Join
                    h_col, p_col, cost_col = mapping['h_name'], mapping['p_name'], mapping['cost_col']
                    
                    df_h['key'] = df_h[h_col].astype(str).str.lower().str.strip()
                    df_p['key'] = df_p[p_col].astype(str).str.lower().str.strip()
                    
                    merged = pd.merge(df_h, df_p, on='key', how='left')
                    
                    # Clean Cost Rate
                    merged[cost_col] = pd.to_numeric(merged[cost_col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
                    merged['Burn'] = merged['Hours'] * merged[cost_col]
                    
                    # Filter Excluded Workstreams
                    task_col = mapping['task_col']
                    clean_data = merged[~merged[task_col].astype(str).str.contains("SEO|Data Science", case=False, na=False)]

                    # Totals
                    revenue = mapping['revenue']
                    burn = clean_data['Burn'].sum()
                    margin = revenue - burn
                    margin_pct = (margin / revenue) * 100 if revenue else 0

                    # --- VISUALS ---
                    st.markdown("---")
                    
                    # CARDS
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Recognized Revenue", f"${revenue:,.0f}")
                    c2.metric("Total Labor Burn", f"${burn:,.0f}", delta=f"{(burn/revenue)*100:.1f}% of Rev", delta_color="inverse")
                    c3.metric("Net Margin %", f"{margin_pct:.1f}%", delta=f"{margin_pct-60:.1f}% vs Target")

                    # CHART
                    st.subheader("Profitability Gap")
                    chart_df = pd.DataFrame({
                        "Metric": ["Revenue", "Labor Burn"],
                        "Amount": [revenue, burn],
                        "Color": ["#6366f1", "#f43f5e"]
                    })
                    fig = px.bar(chart_df, x="Amount", y="Metric", orientation='h', color="Metric", text_auto="$.2s", color_discrete_map={"Revenue": "#6366f1", "Labor Burn": "#f43f5e"})
                    st.plotly_chart(fig, use_container_width=True)

                    # GHOST AUDIT
                    st.subheader("👻 Ghost Team Audit")
                    ghosts = merged[(merged[cost_col] == 0) & (merged['Hours'] > 0)]
                    if not ghosts.empty:
                        st.dataframe(ghosts.groupby(h_col)['Hours'].sum().reset_index().sort_values('Hours', ascending=False))
                    else:
                        st.success("No Ghost Team members found.")
        else:
            st.warning("AI could not confidently identify all 3 files. Please ensure you uploaded Hours, Pricing, and Billing.")
