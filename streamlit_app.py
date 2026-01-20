import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import json
from google import genai
from google.genai import types
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import base64

# Page config
st.set_page_config(
    page_title="DEPT Financial Report Generator",
    page_icon="📊",
    layout="wide"
)

# Initialize Gemini API
@st.cache_resource
def init_gemini():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing Gemini API: {e}")
        st.info("Please add your GEMINI_API_KEY to Streamlit secrets")
        return None

client = init_gemini()

# Helper Functions
def excel_to_base64(file):
    """Convert Excel file to base64 for Gemini API"""
    return base64.b64encode(file.getvalue()).decode()

def interpret_excel_file(file, client):
    """Use Gemini to understand the Excel file structure"""
    
    prompt = f"""Analyze this Excel file and identify:
    1. What type of financial data it contains (CPT Plan / Harvest Hours / Financial Income / Other)
    2. The column headers and their meanings
    3. The structure of the data (which columns are dates, names, hours, rates, amounts, etc.)
    4. Any important metadata or context
    
    Return your analysis as a JSON object with this structure:
    {{
        "file_type": "CPT_PLAN" | "HARVEST_DATA" | "FINANCIAL_INCOME" | "OTHER",
        "description": "brief description of contents",
        "columns": {{
            "column_name": "interpretation"
        }},
        "date_columns": ["list", "of", "date", "columns"],
        "key_metrics": ["hours", "rate", "cost", etc],
        "team_members": ["if identifiable"],
        "date_range": "if identifiable"
    }}
    
    Be thorough and precise."""
    
    try:
        # Read first sheet to show to AI
        df = pd.read_excel(file, sheet_name=0, nrows=10)
        file_preview = df.to_string()
        
        full_prompt = f"{prompt}\n\nFile preview (first 10 rows):\n{file_preview}"
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=full_prompt
        )
        
        # Parse JSON from response
        response_text = response.text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        
        return json.loads(response_text)
    except Exception as e:
        st.error(f"Error interpreting file: {e}")
        return None

def process_files(uploaded_files, client):
    """Process all uploaded files and extract structured data"""
    
    processed_data = {
        "cpt_plan": None,
        "harvest_data": None,
        "financial_income": None,
        "interpretations": []
    }
    
    for file in uploaded_files:
        st.write(f"🔍 Analyzing: **{file.name}**")
        
        # Interpret file
        interpretation = interpret_excel_file(file, client)
        
        if interpretation:
            processed_data["interpretations"].append({
                "filename": file.name,
                "interpretation": interpretation
            })
            
            # Read the actual data
            df = pd.read_excel(file)
            
            # Store based on type
            file_type = interpretation.get("file_type", "OTHER")
            if file_type == "CPT_PLAN":
                processed_data["cpt_plan"] = df
            elif file_type == "HARVEST_DATA":
                processed_data["harvest_data"] = df
            elif file_type == "FINANCIAL_INCOME":
                processed_data["financial_income"] = df
            
            st.success(f"✓ Identified as: {file_type}")
        
        file.seek(0)  # Reset file pointer
    
    return processed_data

def calculate_correlations(hours_data, financial_data):
    """Calculate correlation between hours and financial metrics"""
    
    try:
        # Merge datasets
        merged = pd.merge(
            hours_data[['month', 'total_hours']],
            financial_data[['month', 'burned_value', 'income']],
            on='month'
        )
        
        # Calculate variance
        merged['hours_variance'] = merged['total_hours'] - merged['total_hours'].mean()
        merged['financial_delta'] = merged['burned_value'] - merged['income']
        
        # Correlations
        corr_absolute = np.corrcoef(
            merged['total_hours'].abs(),
            merged['financial_delta'].abs()
        )[0, 1]
        
        corr_variance = np.corrcoef(
            merged['hours_variance'],
            merged['financial_delta']
        )[0, 1]
        
        return {
            "hours_vs_burn_absolute": round(corr_absolute, 2),
            "hours_variance_vs_delta": round(corr_variance, 2)
        }
    except Exception as e:
        st.error(f"Correlation calculation error: {e}")
        return None

def generate_financial_analysis(processed_data, client):
    """Use Gemini to generate comprehensive financial analysis"""
    
    # Prepare data summary for AI - convert to string to avoid JSON serialization issues
    def df_to_summary(df):
        if df is None:
            return "Not provided"
        return df.head(20).to_string()
    
    data_context = f"""
CPT PLAN DATA:
{df_to_summary(processed_data["cpt_plan"])}

HARVEST DATA:
{df_to_summary(processed_data["harvest_data"])}

FINANCIAL INCOME DATA:
{df_to_summary(processed_data["financial_income"])}
"""
    
    prompt = f"""You are a senior financial analyst at DEPT agency. Analyze this financial data and generate a comprehensive report.

DATA PROVIDED:
{data_context}

Generate a detailed financial analysis following this structure:

1. EXECUTIVE SUMMARY
   - Financial status (cumulative profit/loss)
   - Revenue efficiency (% of labor value recovered as income)
   - Primary drivers of performance
   - Key risks

2. METHODOLOGY
   - Correlation analysis (hours vs financial burn)
   - Why financial analysis matters beyond hours
   - Case studies showing seniority drift

3. MONTHLY BREAKDOWN
   - Month-by-month profitability
   - Burned labor value vs DEPT income
   - Efficiency percentage
   - Trend analysis

4. TEAM-LEVEL ANALYSIS
   - Hours variance by team
   - Financial impact per team
   - Seniority drift detection
   - Staff swap analysis

5. STRATEGIC RECOMMENDATIONS
   - Contractual adjustments needed
   - Fee structure changes
   - Operational improvements
   - 2026 action plan

Return as structured JSON with sections and insights. Be specific with numbers and percentages."""

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt
        )
        response_text = response.text.strip()
        
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        
        return json.loads(response_text)
    except Exception as e:
        st.error(f"Analysis generation error: {e}")
        return None

def create_profitability_chart(monthly_data):
    """Create monthly profitability visualization"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly_data['month'],
        y=monthly_data['burned_value'],
        name='Burned Labor Value',
        marker_color='#E74C3C'
    ))
    
    fig.add_trace(go.Bar(
        x=monthly_data['month'],
        y=monthly_data['income'],
        name='DEPT Income',
        marker_color='#27AE60'
    ))
    
    fig.update_layout(
        title='Monthly Profitability: Burned Value vs Income',
        xaxis_title='Month',
        yaxis_title='Amount ($)',
        barmode='group',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_efficiency_chart(monthly_data):
    """Create revenue efficiency trend chart"""
    
    monthly_data['efficiency'] = (monthly_data['income'] / monthly_data['burned_value'] * 100).round(0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_data['month'],
        y=monthly_data['efficiency'],
        mode='lines+markers',
        line=dict(color='#3498DB', width=3),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.1)'
    ))
    
    fig.add_hline(y=100, line_dash="dash", line_color="gray", 
                  annotation_text="Break-even (100%)")
    
    fig.update_layout(
        title='Revenue Efficiency Trend (%)',
        xaxis_title='Month',
        yaxis_title='Efficiency (%)',
        template='plotly_white',
        height=400
    )
    
    return fig

# Main App
def main():
    st.title("📊 DEPT Financial Report Generator")
    st.markdown("### AI-Powered Multi-Account Financial Analysis")
    
    st.markdown("""
    **Upload your financial files to generate comprehensive reports including:**
    - Executive Summary with KPIs
    - Correlation & Methodology Analysis
    - Monthly Profitability Breakdown
    - Team-Level Performance Scorecards
    - Strategic Recommendations
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        account_name = st.text_input("Account Name", "Client Account")
        report_period = st.text_input("Report Period", "2025")
        
        st.markdown("---")
        st.markdown("**File Upload Guidelines:**")
        st.info("""
        Upload any combination of:
        - CPT Plan (planned hours/team)
        - Harvest Data (logged hours)
        - Financial Income (monthly revenue)
        
        Files can have any format - AI will interpret them.
        """)
    
    # File Upload
    uploaded_files = st.file_uploader(
        "Upload Excel Files (.xlsx)",
        type=['xlsx'],
        accept_multiple_files=True,
        help="Upload CPT plans, Harvest data, and Financial income files"
    )
    
    if uploaded_files and client:
        
        if st.button("🚀 Generate Financial Report", type="primary"):
            
            with st.spinner("Processing files with AI..."):
                # Process files
                processed_data = process_files(uploaded_files, client)
                
                st.success(f"✓ Processed {len(uploaded_files)} files successfully")
            
            # Show interpretations
            with st.expander("📋 File Interpretations", expanded=False):
                for item in processed_data["interpretations"]:
                    st.markdown(f"**{item['filename']}**")
                    st.json(item['interpretation'])
            
            # Generate analysis
            with st.spinner("Generating financial analysis..."):
                analysis = generate_financial_analysis(processed_data, client)
            
            if analysis:
                # Display Report
                st.markdown("---")
                st.header(f"📑 Financial Report: {account_name} {report_period}")
                
                # Executive Summary
                st.markdown("## 🎯 Executive Summary")
                if "executive_summary" in analysis:
                    summary = analysis["executive_summary"]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Financial Status",
                            summary.get("cumulative_loss", "N/A"),
                            delta=summary.get("trend", ""),
                            delta_color="inverse"
                        )
                    with col2:
                        st.metric(
                            "Revenue Efficiency",
                            summary.get("efficiency", "N/A"),
                        )
                    with col3:
                        st.metric(
                            "Primary Risk",
                            summary.get("risk_level", "N/A")
                        )
                    
                    st.markdown(summary.get("narrative", ""))
                
                # Methodology
                st.markdown("## 🔬 Methodology")
                if "methodology" in analysis:
                    st.markdown(analysis["methodology"].get("explanation", ""))
                    
                    # Correlation table
                    if "correlations" in analysis["methodology"]:
                        st.markdown("### Correlation Analysis")
                        corr_df = pd.DataFrame(analysis["methodology"]["correlations"])
                        st.dataframe(corr_df, use_container_width=True)
                
                # Monthly Breakdown
                st.markdown("## 📅 Monthly Breakdown")
                if "monthly_breakdown" in analysis:
                    monthly_df = pd.DataFrame(analysis["monthly_breakdown"])
                    
                    # Show table
                    st.dataframe(
                        monthly_df.style.format({
                            'burned_value': '${:,.0f}',
                            'income': '${:,.0f}',
                            'efficiency': '{:.0f}%',
                            'profit_loss': '${:,.0f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Charts
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(
                            create_profitability_chart(monthly_df),
                            use_container_width=True
                        )
                    with col2:
                        st.plotly_chart(
                            create_efficiency_chart(monthly_df),
                            use_container_width=True
                        )
                
                # Team Analysis
                st.markdown("## 👥 Team-Level Analysis")
                if "team_analysis" in analysis:
                    team_df = pd.DataFrame(analysis["team_analysis"])
                    st.dataframe(team_df, use_container_width=True)
                
                # Recommendations
                st.markdown("## 💡 Strategic Recommendations")
                if "recommendations" in analysis:
                    for i, rec in enumerate(analysis["recommendations"], 1):
                        st.markdown(f"**{i}. {rec.get('title', '')}**")
                        st.markdown(rec.get('description', ''))
                        st.markdown("")
                
                # Export
                st.markdown("---")
                st.markdown("### 📥 Export Report")
                
                # Generate JSON export
                export_data = {
                    "account": account_name,
                    "period": report_period,
                    "generated": datetime.now().isoformat(),
                    "analysis": analysis,
                    "raw_data": {
                        "files_processed": len(uploaded_files),
                        "interpretations": processed_data["interpretations"]
                    }
                }
                
                json_export = json.dumps(export_data, indent=2)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📄 Download JSON Report",
                        data=json_export,
                        file_name=f"financial_report_{account_name}_{report_period}.json",
                        mime="application/json"
                    )
                with col2:
                    st.info("PDF export coming soon - use Print to PDF from browser")
    
    else:
        st.info("👆 Upload your Excel files to get started")
        
        # Show example
        with st.expander("📖 See Example Report Structure"):
            st.markdown("""
            Your report will include:
            
            **1. Executive Summary**
            - Financial status overview
            - Revenue efficiency metrics
            - Primary performance drivers
            
            **2. Methodology**
            - Correlation analysis (Hours vs Financial Burn)
            - Seniority mix impact analysis
            
            **3. Monthly Breakdown**
            - Burned labor value vs income
            - Month-over-month profitability
            - Efficiency trends
            
            **4. Team-Level Deep Dive**
            - Hours variance by team
            - Financial impact per team
            - Seniority drift detection
            
            **5. Strategic Recommendations**
            - Contractual adjustments
            - Fee structure optimization
            - 2026 action plan
            """)

if __name__ == "__main__":
    main()
