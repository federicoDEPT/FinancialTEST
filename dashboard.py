"""
dashboard.py

Aplicación Streamlit para generar un dashboard financiero interactivo.

Esta aplicación permite al usuario subir uno o varios archivos Excel con
información financiera, procesarlos mediante las funciones definidas en
``financial_report_app.py`` y visualizar los resultados de forma dinámica.

El script obtiene la clave de API de Gemini desde ``st.secrets`` si está
definida bajo la sección ``[gemini]`` y la pasa a ``read_excel_files`` para
habilitar la detección de columnas asistida por IA. Si no se encuentra la
clave, se utiliza únicamente la heurística local.

Requisitos:
    - streamlit
    - pandas
    - plotly
    - financial_report_app (en el mismo directorio o instalable)

Ejecutar con:
    streamlit run dashboard.py
"""

import os
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
# Obtener la clave de API de Gemini desde st.secrets
# -----------------------------------------------------------------------------

def get_gemini_api_key() -> str | None:
    """Obtiene la clave de API de Gemini desde st.secrets si está disponible."""
    api_key = None
    try:
        # Intentar leer la clave desde secrets usando la sección 'gemini'
        api_key = st.secrets["gemini"]["api_key"]
    except Exception:
        # Si falla, intentar leer de una variable de entorno
        api_key = os.environ.get("GEMINI_API_KEY")
    return api_key


# -----------------------------------------------------------------------------
# Configuración de la aplicación
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Dashboard financiero",
    page_icon=":bar_chart:",
    layout="wide"
)

st.title("📊 Dashboard financiero")
st.markdown("Carga uno o varios archivos Excel y revisa las métricas de rendimiento por equipo.")

# Subir archivos
uploaded_files = st.file_uploader(
    "Selecciona archivos Excel", type=["xlsx", "xls"], accept_multiple_files=True
)

if uploaded_files:
    # Guardar archivos temporales y preparar rutas
    temp_paths = []
    for uploaded in uploaded_files:
        temp_path = f"/tmp/{uploaded.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        temp_paths.append(temp_path)

    # Obtener clave de API de Gemini
    gemini_api_key = get_gemini_api_key()

    # Leer y normalizar los datos con la clave si está disponible
    df_all = read_excel_files(temp_paths, api_key=gemini_api_key)

    # Usar Gemini para analizar todo el dataset y generar métricas y resumen
    metrics_text = None
    summary_text = ""
    if gemini_api_key:
        analysis = get_full_analysis_with_gemini(gemini_api_key, df_all)
        df_metrics = analysis.get('metrics') if analysis.get('metrics') is not None else pd.DataFrame()
        summary_text = analysis.get('summary', '')
        # Si Gemini no devolvió métricas, usar heurística como respaldo
        if df_metrics.empty:
            df_metrics = compute_metrics(df_all)
    else:
        # Si no hay API key, usar heurística y avisar al usuario
        df_metrics = compute_metrics(df_all)
        st.warning("No se encontró una clave de API de Gemini. Se utilizarán heurísticas locales para calcular las métricas y no se generará un análisis completo.")

    # Convertir Periodo a cadena para filtros y visualización
    if not df_metrics.empty and 'mes' in df_metrics.columns:
        df_metrics['mes_str'] = df_metrics['mes'].astype(str)
    else:
        df_metrics['mes_str'] = ''

    # Filtros de Sidebar
    st.sidebar.header("Filtros")
    # Equipos únicos
    equipos = st.sidebar.multiselect(
        "Equipo",
        options=sorted(df_metrics['equipo'].dropna().unique()),
        default=sorted(df_metrics['equipo'].dropna().unique())
    )
    # Períodos únicos
    periodos = st.sidebar.multiselect(
        "Mes",
        options=sorted(df_metrics['mes_str'].unique()),
        default=sorted(df_metrics['mes_str'].unique())
    )

    # Aplicar filtros
    df_filtrado = df_metrics[
        (df_metrics['equipo'].isin(equipos)) &
        (df_metrics['mes_str'].isin(periodos))
    ]

    # Mostrar resumen ejecutivo si existe
    if summary_text:
        st.markdown("## Resumen ejecutivo generado por Gemini")
        st.info(summary_text)
    else:
        st.info("No se recibió un resumen de Gemini. Puede deberse a un problema con la API o a que los datos no pudieron ser analizados.")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Horas totales", f"{df_filtrado['horas'].sum():,.1f}")
    col2.metric("Valor quemado", f"${df_filtrado['valor_quemado'].sum():,.2f}")
    col3.metric("Ingresos", f"${df_filtrado['ingreso'].sum():,.2f}")
    # Eficiencia promedio
    eficiencia_prom = df_filtrado['eficiencia'].mean() * 100 if not df_filtrado.empty else 0
    col4.metric("Eficiencia promedio", f"{eficiencia_prom:.1f}%")

    st.markdown("---")

    # Gráfico de eficiencia por mes
    if not df_filtrado.empty:
        fig_eff = px.line(
            df_filtrado,
            x='mes_str',
            y='eficiencia',
            color='equipo',
            title='Eficiencia por equipo y mes',
            labels={'mes_str': 'Mes', 'eficiencia': 'Eficiencia'},
        )
        # Mostrar gráfico. Streamlit recomienda sustituir use_container_width tras 2025-12-31
        st.plotly_chart(fig_eff, width='stretch')

    # Tabla de datos
    st.subheader("Tabla de métricas")
    st.dataframe(df_filtrado)

    # Opción de descargar el informe en PDF
    if not df_metrics.empty:
        # Ruta temporal para el PDF
        pdf_path = "/tmp/informe_financiero.pdf"
        # Generar PDF con el resumen (si hay)
        generate_financial_report(df_metrics, pdf_path, company_name="Informe", summary=summary_text)
        # Leer el PDF en modo binario
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="Descargar informe en PDF",
            data=pdf_bytes,
            file_name="informe_financiero.pdf",
            mime="application/pdf"
        )

else:
    st.info("Sube uno o varios archivos Excel para comenzar.")
