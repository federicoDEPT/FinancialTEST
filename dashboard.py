"""
dashboard.py

Streamlit application for generating an interactive financial dashboard.

This app allows a user to upload one or more Excel files containing financial
information, process them using the functions defined in
``financial_report_app.py`` and visualise the results dynamically. Optionally,
it can use the Gemini API to detect column labels and produce a full
analysis and executive summary.

The script obtains the Gemini API key from ``st.secrets`` if available
(either ``secrets["gemini"]["api_key"]`` or ``secrets["gemini_api_key"]``) and
passes it to ``read_excel_files`` and analysis functions to enable AI-assisted
processing. If no key is found, local heuristics are used instead.

Requirements:
    - streamlit
    - pandas
    - plotly
    - financial_report_app (in the same directory or installed)

Run with:
    streamlit run dashboard.py
"""

import os
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.express as px

from financial_report_app import (
    read_excel_files,
    compute_metrics,
    generate_summary_with_gemini,
    generate_financial_report,
    get_full_analysis_with_gemini,
)


# -----------------------------------------------------------------------------
# Retrieve the Gemini API key from st.secrets
# -----------------------------------------------------------------------------

def get_gemini_api_key() -> Optional[str]:
    """
    Retrieve the Gemini API key from Streamlit's secret configuration.

    When deploying the application on Streamlit Cloud or another environment,
    you should store your Gemini API key under ``[gemini].api_key`` in the
    secrets file (``.streamlit/secrets.toml``). For backward compatibility
    this helper also checks for a top‑level ``gemini_api_key`` entry or
    falls back to an environment variable named ``GEMINI_API_KEY``. If no
    key is found, the application will run using heuristic header detection
    and will not perform AI‑assisted analysis.

    Returns
    -------
    str or None
        The API key if found, otherwise ``None``.
    """
    api_key: Optional[str] = None
    # First look for a nested ``gemini`` section with ``api_key``
    try:
        api_key = st.secrets.get("gemini", {}).get("api_key")
    except Exception:
        api_key = None
    # Fall back to a top-level secret named ``gemini_api_key`` if present
    if not api_key:
        try:
            api_key = st.secrets.get("gemini_api_key")
        except Exception:
            api_key = None
    # Finally, fall back to the environment variable ``GEMINI_API_KEY``
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    return api_key


# -----------------------------------------------------------------------------
# Application configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Financial Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("📊 Financial Dashboard")
st.markdown(
    "Upload one or more Excel files and review performance metrics by team."
)

# Upload files
uploaded_files = st.file_uploader(
    "Select Excel files", type=["xlsx", "xls"], accept_multiple_files=True
)

if uploaded_files:
    # Guardar archivos temporales y preparar rutas
    temp_paths = []
    for uploaded in uploaded_files:
        temp_path = f"/tmp/{uploaded.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        temp_paths.append(temp_path)

    # Get the Gemini API key
    gemini_api_key = get_gemini_api_key()

    # Read and normalise data; use the API key if available
    df_all = read_excel_files(temp_paths, api_key=gemini_api_key)

    # Use Gemini to analyse the full dataset and generate metrics and summary
    summary_text = ""
    if gemini_api_key:
        analysis = get_full_analysis_with_gemini(gemini_api_key, df_all)
        df_metrics = analysis.get('metrics') if analysis.get('metrics') is not None else pd.DataFrame()
        summary_text = analysis.get('summary', '')
        # If Gemini did not return metrics, fall back to local computation
        if df_metrics.empty:
            df_metrics = compute_metrics(df_all)
    else:
        # If there is no API key, use heuristics and warn the user
        df_metrics = compute_metrics(df_all)
        st.warning(
            "No Gemini API key was found. Local heuristics will be used to calculate metrics and no AI‑generated analysis will be included. "
            "To enable the full analysis and executive summary, please add your Gemini API key to your Streamlit secrets (e.g. in `.streamlit/secrets.toml` under `[gemini].api_key`)."
        )

    # Convert the period to string for filters and visualisation
    if not df_metrics.empty:
        if 'month' in df_metrics.columns:
            df_metrics['month_str'] = df_metrics['month'].astype(str)
        elif 'mes' in df_metrics.columns:
            df_metrics['month_str'] = df_metrics['mes'].astype(str)
        else:
            df_metrics['month_str'] = ''
    else:
        df_metrics['month_str'] = ''

    # Sidebar filters
    st.sidebar.header("Filters")
    # Unique teams
    team_col = 'team' if 'team' in df_metrics.columns else 'equipo'
    teams = st.sidebar.multiselect(
        "Team",
        options=sorted(df_metrics[team_col].dropna().unique()),
        default=sorted(df_metrics[team_col].dropna().unique()),
    )
    # Unique periods
    periods = st.sidebar.multiselect(
        "Month",
        options=sorted(df_metrics['month_str'].unique()),
        default=sorted(df_metrics['month_str'].unique()),
    )

    # Apply filters
    df_filtrado = df_metrics[
        (df_metrics[team_col].isin(teams)) &
        (df_metrics['month_str'].isin(periods))
    ]

    # Show executive summary if available
    if summary_text:
        st.markdown("## Executive summary generated by Gemini")
        st.info(summary_text)
    else:
        st.info(
            "No summary was received from Gemini. This may be due to a problem with the API or because the data could not be analysed."
        )

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    # Determine numeric column names (support both English and Spanish for compatibility)
    hours_col = 'hours' if 'hours' in df_filtrado.columns else 'horas'
    burned_col = 'burned_value' if 'burned_value' in df_filtrado.columns else 'valor_quemado'
    revenue_col = 'revenue' if 'revenue' in df_filtrado.columns else 'ingreso'
    efficiency_col = 'efficiency' if 'efficiency' in df_filtrado.columns else 'eficiencia'
    col1.metric("Total hours", f"{df_filtrado[hours_col].sum():,.1f}")
    col2.metric("Burned value", f"${df_filtrado[burned_col].sum():,.2f}")
    col3.metric("Revenue", f"${df_filtrado[revenue_col].sum():,.2f}")
    # Average efficiency
    avg_efficiency = df_filtrado[efficiency_col].mean() * 100 if not df_filtrado.empty else 0
    col4.metric("Average efficiency", f"{avg_efficiency:.1f}%")

    st.markdown("---")

    # Efficiency plot by month
    if not df_filtrado.empty:
        fig_eff = px.line(
            df_filtrado,
            x='month_str',
            y=efficiency_col,
            color=team_col,
            title='Efficiency by team and month',
            labels={'month_str': 'Month', efficiency_col: 'Efficiency'},
        )
        # Display the chart using a responsive width
        st.plotly_chart(fig_eff, width='stretch')

    # Data table
    st.subheader("Metrics table")
    st.dataframe(df_filtrado)

    # Option to download the report as PDF
    if not df_metrics.empty:
        pdf_path = "/tmp/financial_report.pdf"
        # Create a copy and rename Spanish columns to English equivalents if necessary
        df_pdf = df_metrics.copy()
        rename_map = {
            'mes': 'month',
            'equipo': 'team',
            'horas': 'hours',
            'valor_quemado': 'burned_value',
            'ingreso': 'revenue',
            'eficiencia': 'efficiency',
            'ganancia_perdida': 'profit_loss'
        }
        for old, new in rename_map.items():
            if old in df_pdf.columns:
                df_pdf.rename(columns={old: new}, inplace=True)
        # Generate PDF with the summary (if any)
        generate_financial_report(df_pdf, pdf_path, company_name="Report", summary=summary_text)
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="Download report as PDF",
            data=pdf_bytes,
            file_name="financial_report.pdf",
            mime="application/pdf",
        )

else:
    st.info("Upload one or more Excel files to get started.")
