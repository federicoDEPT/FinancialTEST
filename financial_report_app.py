"""
financial_report_app.py

Este módulo permite:
- Leer múltiples archivos Excel con estructuras heterogéneas.
- Detectar y normalizar columnas principales (fecha, horas, tarifa, ingreso, equipo) mediante heurísticas simples o la API de Gemini (opcional).
- Calcular métricas financieras mensuales por equipo: valor quemado, ingresos, eficiencia y ganancia/pérdida.
- Generar un informe PDF con tablas de resultados.

Requisitos: pandas, numpy, fpdf2, openpyxl (para leer Excel), requests (si se usa Gemini).
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
from pathlib import Path
from fpdf import FPDF

try:
    import requests  # opcional
except ImportError:
    requests = None  # si no está instalado, no se usa

# -----------------------------------------------------------------------------
# Funciones de IA opcional (Gemini)
# -----------------------------------------------------------------------------

def annotate_columns_with_gemini(api_key: str, sample_row: List) -> str:
    """Ejemplo de función para llamar a la API de Gemini.

    Esta función envía un prompt que pide identificar el rol probable de cada
    columna a partir de los valores de una fila de ejemplo. La respuesta debe
    analizarse posteriormente para mapear las columnas. Si requests no está
    disponible o el api_key es None, se devuelve una cadena vacía.
    """
    if requests is None or api_key is None:
        return ""

    prompt = (
        "Estas son las celdas de la primera fila de un archivo Excel. "
        "Identifica el rol probable de cada columna (fecha, horas, tarifa, ingreso, equipo, etc.) "
        "y devuelve una lista con las etiquetas correspondientes: "
        + str(sample_row)
    )
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 256}
    }
    params = {"key": api_key}
    resp = requests.post(url, json=payload, params=params, headers=headers)
    resp.raise_for_status()
    try:
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return resp.text


# -----------------------------------------------------------------------------
# Detección de encabezados y normalización
# -----------------------------------------------------------------------------

def detect_headers(df: pd.DataFrame, api_key: str = None) -> Dict[int, str]:
    """
    Intenta asignar nombres estándar a las columnas de un DataFrame sin encabezados.
    Devuelve un diccionario que mapea índices de columna a nombres estándar.

    Heurísticas empleadas:
    - Si la columna contiene fechas, se asigna 'fecha'.
    - Si la columna contiene números y su media < 50, se asigna 'horas'.
    - Si la media está entre 50 y 500, se asigna 'tarifa'.
    - Si la media es mayor, se asigna 'ingreso'.
    - De lo contrario, se asigna 'equipo'.

    Si se proporciona api_key y se desea usar Gemini, se obtiene una sugerencia
    adicional (no implementada el parseo en este ejemplo).
    """
    mapping: Dict[int, str] = {}
    for idx, col in enumerate(df.columns):
        series = df[col].dropna().head(20)
        # Intentar convertir a datetime
        try:
            # Si al menos el 30% de las primeras 10 celdas se convierten a fechas
            parsed = pd.to_datetime(series, errors='coerce')
            if parsed.notna().sum() >= max(1, int(0.3 * len(series))):
                mapping[idx] = 'fecha'
                continue
        except Exception:
            pass
        # Si es numérica
        if pd.api.types.is_numeric_dtype(series):
            mean_val = series.astype(float).mean()
            if mean_val < 50:
                mapping[idx] = 'horas'
            elif mean_val < 500:
                mapping[idx] = 'tarifa'
            else:
                mapping[idx] = 'ingreso'
        else:
            mapping[idx] = 'equipo'

    # Llamada opcional a Gemini (sin parseo de respuesta en este ejemplo)
    if api_key:
        row_sample = df.iloc[0].tolist()
        _ = annotate_columns_with_gemini(api_key, row_sample)
        # Se podría analizar la respuesta y actualizar mapping
    return mapping


def normalize_dataframe(df: pd.DataFrame, header_map: Dict[int, str]) -> pd.DataFrame:
    """Renombra las columnas según header_map y devuelve solo las estándar."""
    # Crear un diccionario de renombrado basado en índices
    rename_dict = {df.columns[idx]: std_name for idx, std_name in header_map.items()}
    df = df.rename(columns=rename_dict)
    desired_cols = ['fecha', 'equipo', 'horas', 'tarifa', 'ingreso']
    # Añadir columnas faltantes
    for col in desired_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[desired_cols]


def read_excel_files(paths: List[str], api_key: str = None) -> pd.DataFrame:
    """
    Lee múltiples archivos Excel y concatena sus datos normalizados.

    Parameters:
        paths: lista de rutas a archivos Excel.
        api_key: opcional, clave de API para usar Gemini.

    Returns:
        DataFrame con columnas estandarizadas.
    """
    frames = []
    for path in paths:
        xls = pd.ExcelFile(path)
        for sheet_name in xls.sheet_names:
            df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            header_map = detect_headers(df_raw, api_key)
            df_norm = normalize_dataframe(df_raw, header_map)
            frames.append(df_norm)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=['fecha', 'equipo', 'horas', 'tarifa', 'ingreso'])


# -----------------------------------------------------------------------------
# Cálculo de métricas financieras
# -----------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas financieras mensuales por equipo.

    - Convierte 'fecha' a tipo datetime y extrae el período mensual.
    - Calcula el valor quemado = horas × tarifa.
    - Agrupa por mes y equipo: suma horas, valor quemado e ingresos.
    - Calcula eficiencia e ingreso menos valor quemado.
    """
    df = df.copy()
    # Convertir fechas
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df['mes'] = df['fecha'].dt.to_period('M')
    # Calcular valor quemado
    df['horas'] = pd.to_numeric(df['horas'], errors='coerce')
    df['tarifa'] = pd.to_numeric(df['tarifa'], errors='coerce')
    df['ingreso'] = pd.to_numeric(df['ingreso'], errors='coerce')
    df['valor_quemado'] = df['horas'] * df['tarifa']
    # Agrupar
    agg = df.groupby(['mes', 'equipo']).agg({
        'horas': 'sum',
        'valor_quemado': 'sum',
        'ingreso': 'sum'
    }).reset_index()
    # Calcular eficiencia y ganancia/pérdida
    agg['eficiencia'] = np.where(agg['valor_quemado'] != 0,
                                 agg['ingreso'] / agg['valor_quemado'], np.nan)
    agg['ganancia_perdida'] = agg['ingreso'] - agg['valor_quemado']
    return agg


# -----------------------------------------------------------------------------
# Generación de PDF
# -----------------------------------------------------------------------------
class FinancialPDF(FPDF):
    """
    Clase que extiende FPDF para añadir pies de página con numeración.
    """
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')


def generate_financial_report(df_metrics: pd.DataFrame, output_path: str, company_name: str = "Informe"):
    """
    Genera un PDF simple con los resultados en df_metrics.

    Parameters:
        df_metrics: DataFrame con columnas ['mes','equipo','horas','valor_quemado','ingreso','eficiencia','ganancia_perdida'].
        output_path: ruta donde se guardará el PDF.
        company_name: nombre de la organización o proyecto.
    """
    pdf = FinancialPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Título
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f'INFORME FINANCIERO - {company_name}', ln=True)
    pdf.ln(4)
    # Resumen
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, (
        "Este informe presenta la evolución de las horas trabajadas, el valor de trabajo "
        "quemado, los ingresos y la eficiencia de cada equipo por mes. Los valores se "
        "calculan automáticamente a partir de los archivos Excel cargados."))
    pdf.ln(8)

    # Encabezados de tabla
    pdf.set_font('Arial', 'B', 10)
    headers = ['Mes', 'Equipo', 'Horas', 'Valor quemado', 'Ingreso', 'Eficiencia', 'Ganancia/Pérdida']
    col_widths = [25, 35, 25, 35, 30, 25, 40]
    for header, width in zip(headers, col_widths):
        pdf.cell(width, 8, header, 1, 0, 'C')
    pdf.ln()

    # Filas de la tabla
    pdf.set_font('Arial', '', 9)
    for _, row in df_metrics.iterrows():
        pdf.cell(col_widths[0], 7, str(row['mes']), 1)
        pdf.cell(col_widths[1], 7, str(row['equipo']), 1)
        pdf.cell(col_widths[2], 7, f"{row['horas']:.1f}", 1, 0, 'R')
        pdf.cell(col_widths[3], 7, f"${row['valor_quemado']:.2f}", 1, 0, 'R')
        pdf.cell(col_widths[4], 7, f"${row['ingreso']:.2f}", 1, 0, 'R')
        # Eficiencia en %
        eff = row['eficiencia'] * 100 if pd.notna(row['eficiencia']) else 0
        pdf.cell(col_widths[5], 7, f"{eff:.1f}%", 1, 0, 'R')
        pdf.cell(col_widths[6], 7, f"${row['ganancia_perdida']:.2f}", 1, 0, 'R')
        pdf.ln()

    # Guardar PDF
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf.output(output_path)


# -----------------------------------------------------------------------------
# Ejecutable de ejemplo
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generador de informes financieros a partir de archivos Excel.')
    parser.add_argument('files', nargs='+', help='Rutas de archivos Excel a procesar')
    parser.add_argument('-o', '--output', default='informe_financiero.pdf', help='Ruta de salida del PDF')
    parser.add_argument('--api-key', help='Clave de API de Gemini (opcional)')
    parser.add_argument('--company', default='Proyecto', help='Nombre de la empresa o proyecto')
    args = parser.parse_args()

    # Leer y normalizar
    df_all = read_excel_files(args.files, api_key=args.api_key)
    # Calcular métricas
    df_metrics = compute_metrics(df_all)
    # Generar PDF
    generate_financial_report(df_metrics, args.output, company_name=args.company)
    print(f'Informe generado en {args.output}')
