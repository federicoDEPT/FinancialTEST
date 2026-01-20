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
import json
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

    # Construir un prompt estructurado para obtener una lista de etiquetas en orden.
    # Se solicita explícitamente una estructura JSON para facilitar el parseo de la respuesta.
    prompt = (
        "Estas son las celdas de la primera fila de un archivo Excel. "
        "Devuelve un JSON con la clave 'labels' que contenga una lista de etiquetas estandarizadas "
        "para cada columna en el mismo orden. Las etiquetas válidas son 'fecha', 'equipo', 'horas', 'tarifa', 'ingreso'.\n"
        "Ejemplo de respuesta: {\"labels\": [\"fecha\", \"equipo\", \"horas\", \"tarifa\", \"ingreso\"]}.\n"
        "Celdas: " + str(sample_row)
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
# Utilidades para interpretar la respuesta de Gemini y generar resúmenes
# -----------------------------------------------------------------------------
import re

def parse_gemini_mapping(response: str, n_cols: int) -> List[str]:
    """
    Intenta interpretar la respuesta de Gemini para obtener una lista de etiquetas. El
    modelo debería devolver un JSON con la clave 'labels', pero si devuelve texto
    libre o una lista simple, se usan expresiones regulares para extraer las
    etiquetas válidas.

    Parameters
    ----------
    response : str
        Texto devuelto por la API de Gemini.
    n_cols : int
        Número de columnas en el DataFrame original.

    Returns
    -------
    List[str]
        Lista de etiquetas interpretadas. Si no se pueden extraer correctamente,
        se devuelve una lista vacía.
    """
    if not response:
        return []
    text = response.strip()
    # Intentar parsear como JSON para facilitar la extracción
    try:
        import json
        data = json.loads(text)
        if isinstance(data, dict) and 'labels' in data:
            labels_raw = data['labels']
        elif isinstance(data, list):
            labels_raw = data
        else:
            labels_raw = []
        valid_labels = []
        for item in labels_raw:
            if isinstance(item, str):
                low = item.strip().lower()
                if low in ['fecha', 'equipo', 'horas', 'tarifa', 'ingreso']:
                    valid_labels.append(low)
        if valid_labels:
            return valid_labels
    except Exception:
        pass
    # Si no se pudo parsear JSON, utilizar expresiones regulares como respaldo
    keywords = ["fecha", "horas", "tarifa", "ingreso", "equipo"]
    labels: List[str] = []
    for word in re.findall(r"\b\w+\b", text.lower()):
        if word in keywords and word not in labels:
            labels.append(word)
        if len(labels) >= n_cols:
            break
    return labels

def generate_summary_with_gemini(api_key: str, df_metrics: pd.DataFrame) -> str:
    """
    Genera un resumen ejecutivo a partir del DataFrame de métricas mediante la API
    de Gemini. El prompt describe la estructura del DataFrame y pide insights
    clave, tendencias y recomendaciones. Si no se proporciona api_key, se
    devolverá una cadena vacía.

    Parameters
    ----------
    api_key : str
        Clave de API para Gemini.
    df_metrics : pd.DataFrame
        DataFrame con columnas ['mes','equipo','horas','valor_quemado','ingreso','eficiencia','ganancia_perdida'].

    Returns
    -------
    str
        Resumen generado por Gemini o cadena vacía si falla la llamada.
    """
    if requests is None or api_key is None or df_metrics.empty:
        return ""
    # Convertir las métricas a una cadena de tabla simplificada
    table_lines = ["mes\tequipo\thoras\tvalor_quemado\tingreso\teficiencia\tganancia_perdida"]
    for _, row in df_metrics.iterrows():
        line = f"{row['mes']}\t{row['equipo']}\t{row['horas']:.1f}\t{row['valor_quemado']:.2f}\t{row['ingreso']:.2f}\t{row['eficiencia']:.2f}\t{row['ganancia_perdida']:.2f}"
        table_lines.append(line)
    table_text = "\n".join(table_lines)
    prompt = (
        "Eres un analista financiero experto. A continuación se presentan métricas mensuales por equipo en formato de tabla con las columnas: "
        "mes (periodo mensual), equipo, horas totales, valor quemado (horas×tarifa), ingreso total, eficiencia (ingreso/valor quemado) y ganancia o pérdida. "
        "Debes analizar exhaustivamente estas métricas como si estuvieras en un chat de Gemini analizando los archivos originales: "
        "calcula y comenta correlaciones entre las horas y el impacto financiero, detecta cambios en la mezcla de senioridad (por ejemplo, variaciones en la tarifa promedio), "
        "identifica meses con sobreconsumo de horas o alta eficiencia, apunta cualquier riesgo de margen y sugiere acciones correctivas. "
        "Redacta un resumen ejecutivo claro y recomendaciones concretas para cada equipo y periodo. No devuelvas la tabla ni repitas los números textualmente, enfócate en el análisis y las conclusiones.\n\n"
        + table_text
    )
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512}
    }
    params = {"key": api_key}
    try:
        resp = requests.post(url, json=payload, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return ""


def get_full_analysis_with_gemini(api_key: str, df: pd.DataFrame) -> Dict[str, object]:
    """
    Envía el contenido completo del DataFrame a Gemini en formato CSV y pide
    que identifique las columnas relevantes, calcule métricas mensuales por
    equipo (horas, valor quemado, ingreso, eficiencia y ganancia/pérdida) y
    devuelva tanto las métricas como un resumen ejecutivo. El modelo debe
    responder con un JSON que contenga dos claves: "metrics" (lista de
    objetos) y "summary" (cadena).

    Nota: Los modelos generativos tienen un límite de tokens. Para evitar
    excederlo, se limita el número de filas enviadas. Si el DataFrame tiene
    más de 200 filas, se envían solo las primeras 200.

    Parameters
    ----------
    api_key : str
        Clave de API de Gemini.
    df : pd.DataFrame
        DataFrame con los datos originales del Excel (antes de normalizar).

    Returns
    -------
    dict
        Diccionario con dos claves:
        - 'metrics': DataFrame con las métricas calculadas por Gemini o vacío si falla la llamada.
        - 'summary': cadena con el resumen ejecutivo.
    """
    result = {'metrics': pd.DataFrame(), 'summary': ''}
    if requests is None or api_key is None or df.empty:
        return result
    # Limitar el número de filas para evitar prompts demasiado grandes
    max_rows = 200
    if len(df) > max_rows:
        df_to_send = df.head(max_rows).copy()
    else:
        df_to_send = df.copy()
    # Convertir a CSV sin índice
    csv_data = df_to_send.to_csv(index=False)
    # Construir prompt solicitando JSON con métricas y resumen
    prompt = (
        "Eres un analista financiero. A continuación se presenta el contenido de un archivo Excel en formato CSV. "
        "Debes identificar las columnas correspondientes a fecha, equipo, horas, tarifa e ingreso, "
        "agrupando los datos por mes y equipo para calcular las siguientes métricas: horas totales, valor quemado (horas×tarifa), ingreso total, eficiencia (ingreso/valor quemado) y ganancia o pérdida (ingreso - valor quemado). "
        "Devuelve la respuesta en formato JSON con dos claves: 'metrics' y 'summary'. 'metrics' debe ser una lista de objetos con las claves 'mes', 'equipo', 'horas', 'valor_quemado', 'ingreso', 'eficiencia' y 'ganancia_perdida'. 'summary' debe ser un texto que resuma los hallazgos y recomendaciones.\n\n"
        + "CSV:\n" + csv_data
    )
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048}
    }
    params = {"key": api_key}
    try:
        resp = requests.post(url, json=payload, params=params, headers=headers)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        # Intentar parsear como JSON
        data = json.loads(text)
        # Convertir metrics a DataFrame si corresponde
        metrics_list = data.get('metrics') if isinstance(data, dict) else None
        if isinstance(metrics_list, list):
            df_metrics = pd.DataFrame(metrics_list)
            # Asegurar que columnas numéricas se conviertan correctamente
            for col in ['horas', 'valor_quemado', 'ingreso', 'eficiencia', 'ganancia_perdida']:
                if col in df_metrics.columns:
                    df_metrics[col] = pd.to_numeric(df_metrics[col], errors='coerce')
            # Convertir 'mes' a period (si es string)
            if 'mes' in df_metrics.columns:
                df_metrics['mes'] = df_metrics['mes'].astype(str)
        else:
            df_metrics = pd.DataFrame()
        result['metrics'] = df_metrics
        summary_text = data.get('summary', '') if isinstance(data, dict) else ''
        result['summary'] = summary_text
    except Exception:
        # Si no se puede parsear o falla la llamada, devolver resultado vacío
        result['metrics'] = pd.DataFrame()
        result['summary'] = ''
    return result


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
    """
    Intenta asignar nombres estándar a las columnas de un DataFrame. Si se
    proporciona una API key de Gemini, se consulta primero al modelo para
    obtener sugerencias de etiquetas basadas en la primera fila de datos. Las
    sugerencias se aplican en el orden en que aparecen. Si alguna columna
    permanece sin clasificar, se utilizan heurísticas simples para completar
    el mapeo.
    """
    n_cols = len(df.columns)
    mapping: Dict[int, str] = {}
    # Si tenemos API key, consultar primero a Gemini
    suggestions: List[str] = []
    if api_key:
        row_sample = df.iloc[0].tolist()
        try:
            response = annotate_columns_with_gemini(api_key, row_sample)
            suggestions = parse_gemini_mapping(response, n_cols)
        except Exception:
            suggestions = []
    # Aplicar sugerencias en orden
    for idx, label in enumerate(suggestions):
        mapping[idx] = label
    # Para columnas no mapeadas, usar heurísticas
    for idx, col in enumerate(df.columns):
        if idx in mapping:
            continue
        series = df[col].dropna().head(20)
        # Intentar convertir a datetime
        try:
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
    return mapping


def normalize_dataframe(df: pd.DataFrame, header_map: Dict[int, str]) -> pd.DataFrame:
    """
    Renombra las columnas según ``header_map`` y devuelve solo las columnas estándar.

    Si dos o más columnas se mapean al mismo nombre estándar (por ejemplo, si
    varias columnas se clasifican como "equipo"), pandas permite columnas
    duplicadas, lo que más tarde provoca un ``InvalidIndexError`` al concatenar
    dataframes. Para evitarlo, después del renombrado se eliminan las
    columnas duplicadas conservando la primera aparición.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame sin encabezado leído directamente del Excel.
    header_map : Dict[int, str]
        Mapeo de índice de columna a nombre estándar.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas únicas ['fecha','equipo','horas','tarifa','ingreso'].
    """
    # Renombrar columnas según el mapeo de índices
    rename_dict = {df.columns[idx]: std_name for idx, std_name in header_map.items()}
    df = df.rename(columns=rename_dict)
    # Eliminar columnas duplicadas (conservar la primera)
    df = df.loc[:, ~df.columns.duplicated()]
    # Seleccionar columnas deseadas en orden
    desired_cols = ['fecha', 'equipo', 'horas', 'tarifa', 'ingreso']
    # Añadir columnas faltantes con NaN
    for col in desired_cols:
        if col not in df.columns:
            df[col] = np.nan
    # Devuelve solo las columnas deseadas, en el orden especificado
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
            # Leer la hoja sin encabezados
            df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)

            # No hacemos detección de sinónimos; siempre utilizamos Gemini (o heurísticas de backup) para detectar encabezados
            header_map = detect_headers(df_raw, api_key)
            df_norm = normalize_dataframe(df_raw, header_map)
            frames.append(df_norm)
    if frames:
        try:
            # Intentar concatenar directamente
            return pd.concat(frames, ignore_index=True)
        except Exception as e:
            # Puede fallar si existen índices o columnas duplicadas. Aplicar
            # limpieza de duplicados y usar métodos de deduplicación.
            try:
                frames_clean = []
                for df in frames:
                    # Asegurar que las columnas sean únicas dentro de cada DataFrame
                    df = df.loc[:, ~df.columns.duplicated()]
                    frames_clean.append(df)
                return pd.concat(frames_clean, ignore_index=True)
            except Exception:
                # Último recurso: unir con todas las columnas y deduplicar nombres
                concat_df = pd.concat(frames, ignore_index=True, join='outer', sort=False)
                # Deduplicar nombres de columnas añadiendo sufijos
                try:
                    from pandas.io.parsers import ParserBase
                    parser = ParserBase({'names': list(concat_df.columns)})
                    concat_df.columns = parser._maybe_dedup_names(concat_df.columns)
                except Exception:
                    # Si falla la deduplicación, dejar las columnas tal cual
                    pass
                return concat_df
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


# Helper function to sanitize text for PDF
def _sanitize_text(text: str) -> str:
    """
    Replace characters that are not supported by the default FPDF core fonts
    (latin-1 encoding). Converts curly quotes, long dashes and bullets to
    simpler equivalents.

    Parameters
    ----------
    text : str
        Input string to sanitize.

    Returns
    -------
    str
        Sanitized string safe for FPDF.
    """
    if not text:
        return ""
    replacements = {
        '’': "'",
        '‘': "'",
        '“': '"',
        '”': '"',
        '–': '-',  # en dash
        '—': '-',  # em dash
        '•': '*',  # bullet
        '·': '-',
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text


def generate_financial_report(
    df_metrics: pd.DataFrame,
    output_path: str,
    company_name: str = "Informe",
    summary: str | None = None,
) -> None:
    """
    Genera un informe PDF con las métricas y un resumen opcional.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        DataFrame con columnas ['mes','equipo','horas','valor_quemado','ingreso','eficiencia','ganancia_perdida'].
    output_path : str
        Ruta donde se guardará el PDF.
    company_name : str, opcional
        Nombre de la organización o proyecto.
    summary : str, opcional
        Resumen ejecutivo que se incluirá al inicio del informe. Si se omite o
        ``None``, se mostrará solo una breve introducción.
    """
    pdf = FinancialPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Título
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f'INFORME FINANCIERO - {company_name}', ln=True)
    pdf.ln(4)
    # Introducción y resumen
    pdf.set_font('Arial', '', 11)
    intro = (
        "Este informe presenta la evolución de las horas trabajadas, el valor de trabajo "
        "quemado, los ingresos y la eficiencia de cada equipo por mes. Los valores se "
        "calculan automáticamente a partir de los archivos Excel cargados."
    )
    pdf.multi_cell(0, 6, _sanitize_text(intro))
    pdf.ln(4)
    if summary:
        # Añadir encabezado para el resumen
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 7, "Resumen ejecutivo", ln=True)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, _sanitize_text(summary.strip()))
        pdf.ln(4)

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
        pdf.cell(col_widths[0], 7, _sanitize_text(str(row['mes'])), 1)
        pdf.cell(col_widths[1], 7, _sanitize_text(str(row['equipo'])), 1)
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
    # Generar resumen con Gemini si se proporciona API key
    summary = None
    if args.api_key:
        try:
            summary = generate_summary_with_gemini(args.api_key, df_metrics)
        except Exception:
            summary = None
    # Generar PDF con resumen opcional
    generate_financial_report(df_metrics, args.output, company_name=args.company, summary=summary)
    print(f'Informe generado en {args.output}')
