import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import json
from google import genai
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

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

def identify_file_type(df, filename, client):
    """Use Gemini to identify what type of file this is"""
    
    # Get sample of data
    sample = df.head(10).to_string()
    columns = ", ".join([str(col) for col in df.columns.tolist()])
    
    prompt = f"""Analyze this Excel file and identify its type.

Filename: {filename}
Columns: {columns}

Sample data:
{sample}

Is this:
A) CPT_PLAN - Contains planned hours, team member names, roles, rates
B) HARVEST_DATA - Contains logged/tracked hours, actual time entries
C) FINANCIAL_INCOME - Contains revenue, income, monthly financial data

Respond with ONLY one word: CPT_PLAN, HARVEST_DATA, or FINANCIAL_INCOME"""

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt
        )
        file_type = response.text.strip().upper()
        return file_type if file_type in ["CPT_PLAN", "HARVEST_DATA", "FINANCIAL_INCOME"] else "OTHER"
    except Exception as e:
        st.warning(f"Could not identify file type: {e}")
        return "OTHER"

def process_cpt_data(df):
    """Extract planned hours and rates from CPT plan"""
    try:
        # Try to find columns with hours and rates
        data = {
            'total_planned_hours': 0,
            'total_planned_cost': 0,
            'team_breakdown': []
        }
        
        # Look for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Sum all numeric values as rough estimate
            for col in numeric_cols:
                col_str = str(col).lower()
                if 'hour' in col_str or 'time' in col_str:
                    data['total_planned_hours'] += df[col].sum()
                if 'rate' in col_str or 'cost' in col_str or 'price' in col_str:
                    data['total_planned_cost'] += df[col].sum()
        
        return data
    except Exception as e:
        st.warning(f"Error processing CPT data: {e}")
        return {'total_planned_hours': 0, 'total_planned_cost': 0, 'team_breakdown': []}

def process_harvest_data(df):
    """Extract actual hours from Harvest data"""
    try:
        data = {
            'total_actual_hours': 0,
            'monthly_hours': {},
            'team_hours': {}
        }
        
        # Find date and hours columns
        date_cols = [col for col in df.columns if 'date' in str(col).lower() or 'month' in str(col).lower()]
        hour_cols = [col for col in df.columns if 'hour' in str(col).lower() or 'time' in str(col).lower()]
        
        if hour_cols:
            data['total_actual_hours'] = df[hour_cols[0]].sum()
        
        # Try to group by month if date column exists
        if date_cols and hour_cols:
            df_copy = df.copy()
            df_copy['date_parsed'] = pd.to_datetime(df_copy[date_cols[0]], errors='coerce')
            df_copy['month'] = df_copy['date_parsed'].dt.to_period('M')
            monthly = df_copy.groupby('month')[hour_cols[0]].sum()
            data['monthly_hours'] = {str(k): float(v) for k, v in monthly.items()}
        
        return data
    except Exception as e:
        st.warning(f"Error processing Harvest data: {e}")
        return {'total_actual_hours': 0, 'monthly_hours': {}, 'team_hours': {}}

def process_financial_data(df):
    """Extract revenue and income data"""
    try:
        data = {
            'total_income': 0,
            'monthly_income': {},
            'total_cost': 0
        }
        
        # Find relevant columns
        income_cols = [col for col in df.columns if 'income' in str(col).lower() or 'revenue' in str(col).lower()]
        cost_cols = [col for col in df.columns if 'cost' in str(col).lower() or 'expense' in str(col).lower() or 'burn' in str(col).lower()]
        
        if income_cols:
            data['total_income'] = float(df[income_cols[0]].sum())
        if cost_cols:
            data['total_cost'] = float(df[cost_cols[0]].sum())
        
        return data
    except Exception as e:
        st.warning(f"Error processing financial data: {e}")
        return {'total_income': 0, 'monthly_income': {}, 'total_cost': 0}

def generate_ai_insights(data_summary, client):
    """Generate insights using Gemini"""
    
    prompt = f"""You are a financial analyst at DEPT agency. Analyze this data and provide insights.

DATA:
{data_summary}

Provide 5 key insights about:
1. Overall financial health
2. Efficiency (actual vs planned hours)
3. Revenue performance
4. Key risks
5. Recommendations

Format as a numbered list. Be specific and actionable."""

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error generating insights: {e}"

def create_efficiency_gauge(efficiency_pct):
    """Create a gauge chart for revenue efficiency"""
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = efficiency_pct,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Revenue Efficiency"},
        delta = {'reference': 100},
        gauge = {
            'axis': {'range': [None, 150]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': "#ffcccc"},
                {'range': [60, 90], 'color': "#ffffcc"},
                {'range': [90, 110], 'color': "#ccffcc"},
                {'range': [110, 150], 'color': "#99ff99"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 100
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_variance_chart(planned, actual, label):
    """Create a comparison chart"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[label],
        y=[planned],
        name='Planned',
        marker_color='#3498DB'
    ))
    
    fig.add_trace(go.Bar(
        x=[label],
        y=[actual],
        name='Actual',
        marker_color='#E74C3C'
    ))
    
    fig.update_layout(
        title=f'{label}: Planned vs Actual',
        barmode='group',
        height=300,
        showlegend=True
    )
    
    return fig

# Main App
def main():
    st.title("📊 DEPT Financial Report Generator")
    st.markdown("### AI-Powered Multi-Account Financial Analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        account_name = st.text_input("Account Name", "Client Account")
        report_period = st.text_input("Report Period", "2025")
        
        st.markdown("---")
        st.markdown("**About This Tool:**")
        st.info("""
        Upload your financial files to generate:
        - Executive Summary
        - Financial Health Analysis
        - Team Performance Metrics
        - AI-Powered Insights
        - Strategic Recommendations
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
            
            # Process files
            with st.spinner("🔍 Analyzing files..."):
                
                file_data = {
                    'cpt_plan': None,
                    'harvest_data': None,
                    'financial_income': None
                }
                
                for file in uploaded_files:
                    df = pd.read_excel(file)
                    file_type = identify_file_type(df, file.name, client)
                    
                    st.write(f"**{file.name}**: Identified as `{file_type}`")
                    
                    if file_type == "CPT_PLAN":
                        file_data['cpt_plan'] = df
                    elif file_type == "HARVEST_DATA":
                        file_data['harvest_data'] = df
                    elif file_type == "FINANCIAL_INCOME":
                        file_data['financial_income'] = df
            
            # Extract metrics
            with st.spinner("📊 Extracting metrics..."):
                
                cpt_metrics = process_cpt_data(file_data['cpt_plan']) if file_data['cpt_plan'] is not None else {'total_planned_hours': 0, 'total_planned_cost': 0}
                harvest_metrics = process_harvest_data(file_data['harvest_data']) if file_data['harvest_data'] is not None else {'total_actual_hours': 0, 'monthly_hours': {}}
                financial_metrics = process_financial_data(file_data['financial_income']) if file_data['financial_income'] is not None else {'total_income': 0, 'total_cost': 0}
                
                # Calculate key metrics
                planned_hours = float(cpt_metrics.get('total_planned_hours', 0))
                actual_hours = float(harvest_metrics.get('total_actual_hours', 0))
                planned_cost = float(cpt_metrics.get('total_planned_cost', 0))
                total_income = float(financial_metrics.get('total_income', 0))
                total_cost = float(financial_metrics.get('total_cost', 0))
                
                # Calculate efficiency
                if total_cost > 0:
                    efficiency = (total_income / total_cost * 100)
                else:
                    efficiency = 0
                
                # Hours variance
                if planned_hours > 0:
                    hours_variance = ((actual_hours - planned_hours) / planned_hours * 100)
                else:
                    hours_variance = 0
                
                # Profit/Loss
                profit_loss = total_income - total_cost
            
            # Display Report
            st.markdown("---")
            st.header(f"📑 Financial Report: {account_name} {report_period}")
            
            # Executive Summary
            st.markdown("## 🎯 Executive Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Income",
                    f"${total_income:,.0f}",
                    delta=f"${profit_loss:,.0f}" if profit_loss != 0 else None
                )
            
            with col2:
                st.metric(
                    "Total Cost",
                    f"${total_cost:,.0f}"
                )
            
            with col3:
                st.metric(
                    "Revenue Efficiency",
                    f"{efficiency:.1f}%",
                    delta=f"{efficiency - 100:.1f}%" if efficiency > 0 else None,
                    delta_color="normal" if efficiency >= 100 else "inverse"
                )
            
            with col4:
                st.metric(
                    "Hours Variance",
                    f"{hours_variance:+.1f}%",
                    delta_color="inverse" if hours_variance > 0 else "normal"
                )
            
            # Financial Status
            if profit_loss >= 0:
                st.success(f"✅ **Profitable**: +${profit_loss:,.0f}")
            else:
                st.error(f"⚠️ **Loss**: ${profit_loss:,.0f}")
            
            # Visualizations
            st.markdown("## 📈 Key Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    create_efficiency_gauge(efficiency),
                    use_container_width=True
                )
            
            with col2:
                if planned_hours > 0 and actual_hours > 0:
                    st.plotly_chart(
                        create_variance_chart(planned_hours, actual_hours, "Hours"),
                        use_container_width=True
                    )
            
            # Monthly breakdown if available
            if harvest_metrics.get('monthly_hours'):
                st.markdown("## 📅 Monthly Breakdown")
                
                monthly_data = harvest_metrics['monthly_hours']
                if monthly_data:
                    monthly_df = pd.DataFrame([
                        {'Month': k, 'Hours': v}
                        for k, v in monthly_data.items()
                    ])
                    
                    fig = px.line(
                        monthly_df,
                        x='Month',
                        y='Hours',
                        title='Monthly Hours Trend',
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(monthly_df, use_container_width=True)
            
            # AI Insights
            st.markdown("## 💡 AI-Powered Insights")
            
            with st.spinner("🤖 Generating insights..."):
                data_summary = f"""
                Total Income: ${total_income:,.0f}
                Total Cost: ${total_cost:,.0f}
                Profit/Loss: ${profit_loss:,.0f}
                Revenue Efficiency: {efficiency:.1f}%
                Planned Hours: {planned_hours:,.0f}
                Actual Hours: {actual_hours:,.0f}
                Hours Variance: {hours_variance:+.1f}%
                """
                
                insights = generate_ai_insights(data_summary, client)
                st.markdown(insights)
            
            # Data Tables
            st.markdown("## 📋 Data Tables")
            
            tabs = st.tabs(["CPT Plan", "Harvest Data", "Financial Income"])
            
            with tabs[0]:
                if file_data['cpt_plan'] is not None:
                    st.dataframe(file_data['cpt_plan'], use_container_width=True)
                else:
                    st.info("No CPT Plan data uploaded")
            
            with tabs[1]:
                if file_data['harvest_data'] is not None:
                    st.dataframe(file_data['harvest_data'], use_container_width=True)
                else:
                    st.info("No Harvest data uploaded")
            
            with tabs[2]:
                if file_data['financial_income'] is not None:
                    st.dataframe(file_data['financial_income'], use_container_width=True)
                else:
                    st.info("No Financial Income data uploaded")
            
            # Export
            st.markdown("---")
            st.markdown("### 📥 Export Report")
            
            report_summary = {
                "account": account_name,
                "period": report_period,
                "generated": datetime.now().isoformat(),
                "metrics": {
                    "total_income": float(total_income),
                    "total_cost": float(total_cost),
                    "profit_loss": float(profit_loss),
                    "efficiency": float(efficiency),
                    "planned_hours": float(planned_hours),
                    "actual_hours": float(actual_hours),
                    "hours_variance": float(hours_variance)
                }
            }
            
            st.download_button(
                label="📄 Download JSON Report",
                data=json.dumps(report_summary, indent=2),
                file_name=f"financial_report_{account_name}_{report_period}.json",
                mime="application/json"
            )
    
    else:
        st.info("👆 Upload your Excel files to get started")
        
        # Example
        with st.expander("📖 How It Works"):
            st.markdown("""
            **Step 1:** Upload your financial files
            - CPT Plan (planned hours & rates)
            - Harvest Data (actual logged hours)
            - Financial Income (revenue data)
            
            **Step 2:** AI identifies file types automatically
            
            **Step 3:** Get comprehensive analysis:
            - Revenue efficiency metrics
            - Hours variance analysis
            - Profit/loss calculations
            - AI-generated insights
            - Visual charts and trends
            
            **Step 4:** Export as JSON or print to PDF
            """)

if __name__ == "__main__":
    main()
