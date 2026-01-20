"""
financial_report_app.py

This module provides utilities to build automated financial reports from one or
more heterogeneous Excel files. It can normalise columns, compute monthly
metrics per team, interact with the Gemini API for intelligent analysis and
generate a PDF report.

Features
--------
* Read multiple Excel files regardless of varying structures.
* Detect and normalise the key columns (date, team, hours, rate, revenue)
  using simple heuristics or, optionally, the Gemini API.
* Compute monthly financial metrics per team: total hours, burned value,
  revenue, efficiency and profit/loss.
* Generate a PDF report summarising the metrics with an optional
  executive summary provided by Gemini.

Dependencies: pandas, numpy, fpdf2, openpyxl (for reading Excel files),
requests (only if using Gemini).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from datetime import datetime
from fpdf import FPDF

# Attempt to import requests; if unavailable, Gemini integration is disabled.
try:
    import requests  # type: ignore
except Exception:
    requests = None

################################################################################
# Optional AI functions (Gemini)
################################################################################

def annotate_columns_with_gemini(api_key: str, sample_row: List) -> str:
    """
    Call the Gemini API to infer column roles.

    This helper sends a structured prompt asking the model to identify the
    likely role of each column based on values from a sample row. The model
    should return a JSON object with a key ``labels`` containing a list of
    standardised labels (``date``, ``team``, ``hours``, ``rate``, ``revenue``)
    in the same order as the input cells. If the ``requests`` library is not
    available or ``api_key`` is ``None``, an empty string is returned.
    """
    if requests is None or not api_key:
        return ""
    prompt = (
        "These are the cells from the first row of an Excel file. "
        "Return a JSON object with a key 'labels' containing a list of standardised column labels, "
        "in the same order as the input cells. The valid labels are 'date', 'team', 'hours', 'rate', 'revenue'.\n"
        "Example response: {\"labels\": [\"date\", \"team\", \"hours\", \"rate\", \"revenue\"]}.\n"
        "Cells: " + str(sample_row)
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
    try:
        resp = requests.post(url, json=payload, params=params, headers=headers)  # type: ignore
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return ""


def parse_gemini_mapping(response: str, n_cols: int) -> List[str]:
    """
    Interpret the response from Gemini and extract a list of column labels.

    The model is expected to return a JSON object with the key ``labels``.
    However, if it returns free-form text or a simple list, this helper will
    attempt to parse out any valid labels using regular expressions. Both
    English and Spanish synonyms are recognised and mapped to English.

    Parameters
    ----------
    response : str
        Raw text returned by the Gemini API.
    n_cols : int
        Number of columns in the original DataFrame.

    Returns
    -------
    List[str]
        A list of interpreted labels. If parsing fails, an empty list is
        returned.
    """
    if not response:
        return []
    text = response.strip()
    # Try to parse as JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict) and 'labels' in data:
            labels_raw = data['labels']
        elif isinstance(data, list):
            labels_raw = data
        else:
            labels_raw = []
        valid_labels: List[str] = []
        for item in labels_raw:
            if isinstance(item, str):
                low = item.strip().lower()
                # Accept both English and Spanish synonyms
                synonyms = {
                    'fecha': 'date',
                    'date': 'date',
                    'equipo': 'team',
                    'team': 'team',
                    'horas': 'hours',
                    'hours': 'hours',
                    'tarifa': 'rate',
                    'rate': 'rate',
                    'ingreso': 'revenue',
                    'revenue': 'revenue'
                }
                if low in synonyms:
                    valid_labels.append(synonyms[low])
        if valid_labels:
            return valid_labels
    except Exception:
        pass
    # Fall back to extracting keywords with regex
    import re
    keywords = ['date', 'team', 'hours', 'rate', 'revenue', 'fecha', 'equipo', 'horas', 'tarifa', 'ingreso']
    labels: List[str] = []
    for word in re.findall(r"\b\w+\b", text.lower()):
        if word in keywords and word not in labels:
            if word in ['fecha', 'equipo', 'horas', 'tarifa', 'ingreso']:
                word_map = {'fecha': 'date', 'equipo': 'team', 'horas': 'hours', 'tarifa': 'rate', 'ingreso': 'revenue'}
                labels.append(word_map[word])
            else:
                labels.append(word)
        if len(labels) >= n_cols:
            break
    return labels


def generate_summary_with_gemini(api_key: str, df_metrics: pd.DataFrame) -> str:
    """
    Generate an executive summary from the metrics DataFrame using the Gemini API.

    The prompt describes the structure of the DataFrame and asks for key
    insights, trends and actionable recommendations. If no ``api_key`` is
    provided or the request fails, an empty string is returned.

    Parameters
    ----------
    api_key : str
        API key for Gemini.
    df_metrics : pd.DataFrame
        DataFrame containing columns ['month','team','hours','burned_value',
        'revenue','efficiency','profit_loss'].

    Returns
    -------
    str
        The summary generated by Gemini or an empty string if the call fails.
    """
    if requests is None or not api_key or df_metrics.empty:
        return ""
    # Convert the metrics to a simplified tab-delimited string to include in the prompt
    table_lines = ["month\tteam\thours\tburned_value\trevenue\tefficiency\tprofit_loss"]
    for _, row in df_metrics.iterrows():
        line = (
            f"{row['month']}\t{row['team']}\t{row['hours']:.1f}\t"
            f"{row['burned_value']:.2f}\t{row['revenue']:.2f}\t"
            f"{row['efficiency']:.2f}\t{row['profit_loss']:.2f}"
        )
        table_lines.append(line)
    table_text = "\n".join(table_lines)
    prompt = (
        "You are a senior financial analyst. Below are monthly metrics by team in a tabular form with the columns: "
        "month (monthly period), team, total hours, burned value (hours × rate), total revenue, efficiency (revenue/burned value) and profit or loss. "
        "Analyse these metrics thoroughly as if you were conversing in a Gemini chat analysing the original files: "
        "calculate and discuss correlations between hours and financial impact, detect changes in the seniority mix (for example, variations in the average rate), "
        "identify months with overconsumption of hours or high efficiency, highlight any margin risks and suggest corrective actions. "
        "Write a clear executive summary in English and provide concrete recommendations for each team and period. "
        "Do not return the table or repeat the numbers verbatim; focus on the analysis and conclusions.\n\n"
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
        resp = requests.post(url, json=payload, params=params, headers=headers)  # type: ignore
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return ""


def get_full_analysis_with_gemini(api_key: str, df: pd.DataFrame) -> Dict[str, object]:
    """
    Send the full DataFrame content to Gemini in CSV format and ask it to identify the relevant columns,
    compute monthly metrics per team (hours, burned value, revenue, efficiency and profit/loss) and return both
    the metrics and an executive summary. The model should respond with a JSON object containing two keys:
    ``metrics`` (list of objects) and ``summary`` (string).

    Note: generative models have a token limit. To avoid exceeding it, the number of rows sent is capped. If
    the DataFrame has more than 200 rows, only the first 200 rows are included in the prompt.

    Parameters
    ----------
    api_key : str
        Gemini API key.
    df : pd.DataFrame
        The original DataFrame read from Excel (before normalisation).

    Returns
    -------
    dict
        Dictionary with two keys:
        - ``metrics``: DataFrame with the metrics calculated by Gemini or empty if the call fails.
        - ``summary``: string containing the executive summary.
    """
    result = {'metrics': pd.DataFrame(), 'summary': ''}
    if requests is None or not api_key or df.empty:
        return result
    # Send the entire DataFrame to Gemini. Previously we limited the
    # request to the first 200 rows to avoid hitting model token limits.
    # To more closely match the behaviour of chatting directly with the AI on
    # unstructured data, we now send all rows. Be aware that very large files
    # may exceed the API's token limits and return an error.
    df_to_send = df.copy()
    # Convert to CSV without index
    csv_data = df_to_send.to_csv(index=False)
    # Build prompt requesting JSON with metrics and summary
    prompt = (
        "You are a financial analyst. The following is the content of an Excel file in CSV format. "
        "Identify which columns correspond to date, team, hours, rate and revenue. Numeric values may include currency symbols (such as $, €, £), thousand separators (commas or periods) or parentheses for negatives; interpret them as numbers. "
        "Group the data by month and team and compute these metrics: total hours, burned value (hours × rate), total revenue, efficiency (revenue divided by burned value) and profit or loss (revenue minus burned value). "
        "Return the response as a JSON object with two keys: 'metrics' and 'summary'. "
        "'metrics' should be a list of objects with the keys 'month', 'team', 'hours', 'burned_value', 'revenue', 'efficiency' and 'profit_loss'. "
        "'summary' should be a text that summarises the findings and recommendations in English.\n\n"
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
        resp = requests.post(url, json=payload, params=params, headers=headers)  # type: ignore
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        # Attempt to parse the response as JSON
        data = json.loads(text)
        metrics_list = data.get('metrics') if isinstance(data, dict) else None
        if isinstance(metrics_list, list):
            df_metrics = pd.DataFrame(metrics_list)
            # Convert numeric columns appropriately, supporting both English and Spanish names
            numeric_cols = ['hours', 'burned_value', 'revenue', 'efficiency', 'profit_loss',
                            'horas', 'valor_quemado', 'ingreso', 'eficiencia', 'ganancia_perdida']
            for col in numeric_cols:
                if col in df_metrics.columns:
                    df_metrics[col] = pd.to_numeric(df_metrics[col], errors='coerce')
            # Ensure 'month' or Spanish 'mes' columns are strings
            if 'month' in df_metrics.columns:
                df_metrics['month'] = df_metrics['month'].astype(str)
            if 'mes' in df_metrics.columns:
                df_metrics['mes'] = df_metrics['mes'].astype(str)
        else:
            df_metrics = pd.DataFrame()
        result['metrics'] = df_metrics
        summary_text = data.get('summary', '') if isinstance(data, dict) else ''
        result['summary'] = summary_text
    except Exception:
        # Return empty result on failure
        result['metrics'] = pd.DataFrame()
        result['summary'] = ''
    return result


################################################################################
# Header detection and normalisation
################################################################################

def detect_headers(df: pd.DataFrame, api_key: Optional[str] = None) -> Dict[int, str]:
    """
    Attempt to assign standard names to the columns of a DataFrame without headers.

    If an API key is provided, the model is consulted first to obtain label
    suggestions based on the first row of data. Suggestions are applied in order;
    for any remaining columns heuristics are used. The heuristics are:

    * If the column contains dates (based on successful parsing), label it as ``date``.
    * If the column contains numbers and its mean is less than 50, label it as ``hours``.
    * If the mean is between 50 and 500, label it as ``rate``.
    * If the mean is greater or equal to 500, label it as ``revenue``.
    * Otherwise, label it as ``team``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyse.
    api_key : str, optional
        Gemini API key used to obtain suggestions. If ``None``, only heuristics
        are used.

    Returns
    -------
    Dict[int, str]
        Mapping from column index to a standard name.
    """
    # If the dataframe is empty (no rows), we cannot sample any values. Return an empty mapping.
    if df.empty or df.shape[0] == 0:
        return {}
    n_cols = len(df.columns)
    mapping: Dict[int, str] = {}
    # If we have an API key, consult Gemini first
    suggestions: List[str] = []
    if api_key:
        # Sample the first row to infer column labels. Protect against index errors.
        try:
            row_sample = df.iloc[0].tolist()
            response = annotate_columns_with_gemini(api_key, row_sample)
            suggestions = parse_gemini_mapping(response, n_cols)
        except Exception:
            suggestions = []
    # Apply suggestions in order
    for idx, label in enumerate(suggestions):
        mapping[idx] = label
    # Use heuristics for unmapped columns
    for idx, col in enumerate(df.columns):
        if idx in mapping:
            continue
        series = df[col].dropna().head(20)
        # Try to interpret as date
        try:
            parsed = pd.to_datetime(series, errors='coerce')
            if parsed.notna().sum() >= max(1, int(0.3 * len(series))):
                mapping[idx] = 'date'
                continue
        except Exception:
            pass
        # Attempt to interpret as numeric by cleaning currency symbols and separators
        # Convert the series to string for cleaning
        s_str = series.astype(str)
        # Remove any characters that are not digits, commas, dots or minus signs
        cleaned = s_str.str.replace(r'[^\d,\.\-]', '', regex=True)
        # Function to unify decimal separators: if there is a single comma and no dot, replace comma with dot
        def unify_decimal(val: str) -> str:
            if val.count(',') == 1 and val.count('.') == 0:
                return val.replace(',', '.')
            return val
        unified = cleaned.apply(unify_decimal)
        # Convert to numeric (coerces errors to NaN)
        numeric_series = pd.to_numeric(unified, errors='coerce')
        # Determine if this should be treated as numeric
        if numeric_series.notna().sum() >= max(1, int(0.3 * len(series))):
            # Compute the mean value of the numeric data (ignoring NaN)
            mean_val = numeric_series.mean()
            # Use heuristics to classify the numeric column based on typical value ranges
            # Smaller means correspond to hours, mid-range to rate, larger to revenue
            if mean_val < 50:
                mapping[idx] = 'hours'
            elif mean_val < 500:
                mapping[idx] = 'rate'
            else:
                mapping[idx] = 'revenue'
        else:
            # If not numeric and not date, treat as team
            mapping[idx] = 'team'
    return mapping


def normalize_dataframe(df: pd.DataFrame, header_map: Dict[int, str]) -> pd.DataFrame:
    """
    Rename columns according to ``header_map`` and return only the standard columns.

    If two or more columns map to the same standard name (for example, if
    multiple columns are classified as ``team``), pandas will allow duplicate
    columns. Concatenating such DataFrames can later raise an ``InvalidIndexError``.
    To avoid this, duplicate columns are removed after renaming, keeping the first
    occurrence.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame without headers read directly from Excel.
    header_map : Dict[int, str]
        Mapping from column index to standard column name.

    Returns
    -------
    pd.DataFrame
        DataFrame with unique columns ['date','team','hours','rate','revenue'] in order.
    """
    # Rename columns according to the index mapping
    rename_dict = {df.columns[idx]: std_name for idx, std_name in header_map.items()}
    df = df.rename(columns=rename_dict)
    # Remove duplicate columns (keep the first)
    df = df.loc[:, ~df.columns.duplicated()]
    # Desired columns in standard order
    desired_cols = ['date', 'team', 'hours', 'rate', 'revenue']
    # Add missing columns as NaN
    for col in desired_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[desired_cols]


def read_excel_files(paths: List[str], api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Read and normalise the contents of multiple Excel files.

    For each file in ``paths`` all worksheets are loaded without assuming any
    header row. Columns are then assigned standard names (``date``, ``team``,
    ``hours``, ``rate``, ``revenue``) via ``detect_headers`` which optionally
    consults Gemini. The raw sheets are normalised to these columns and any
    duplicate column names are dropped. Finally all sheets are concatenated
    together. To avoid the ``InvalidIndexError`` raised by pandas when
    concatenating frames with duplicate column names, the function builds the
    union of columns across all frames and fills missing values with
    ``NaN`` before concatenation.

    Parameters
    ----------
    paths : List[str]
        Paths to Excel files to read.
    api_key : str, optional
        Gemini API key used to improve header detection. If ``None``, only
        heuristics are used.

    Returns
    -------
    pd.DataFrame
        A single DataFrame containing all normalised data from the input files.
    """
    frames: List[pd.DataFrame] = []
    for path in paths:
        xls = pd.ExcelFile(path)
        for sheet_name in xls.sheet_names:
            # Read each sheet without a header row
            df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            # Skip entirely empty sheets
            if df_raw.empty or df_raw.shape[0] == 0:
                continue
            # Detect headers using Gemini (if api_key provided) or heuristics
            header_map = detect_headers(df_raw, api_key)
            # If no headers could be detected and no heuristics apply, skip this sheet
            if not header_map:
                continue
            df_norm = normalize_dataframe(df_raw, header_map)
            # Only append non-empty DataFrames
            if not df_norm.empty:
                frames.append(df_norm)
    if frames:
        # Ensure each DataFrame has unique column names by removing duplicates
        frames_unique: List[pd.DataFrame] = []
        for df in frames:
            df_clean = df.loc[:, ~df.columns.duplicated()].copy()
            frames_unique.append(df_clean)
        # Collect the union of all column names across frames
        all_columns: List[str] = []
        for df in frames_unique:
            for col in df.columns:
                if col not in all_columns:
                    all_columns.append(col)
        # Pad each DataFrame to contain all columns, filling missing values with NA
        standardized_frames: List[pd.DataFrame] = []
        for df in frames_unique:
            for col in all_columns:
                if col not in df.columns:
                    df[col] = pd.NA
            standardized_frames.append(df[all_columns])
        # Concatenate all standardised DataFrames
        return pd.concat(standardized_frames, ignore_index=True)
    # If no frames were produced, return an empty DataFrame with the standard columns
    return pd.DataFrame(columns=['date', 'team', 'hours', 'rate', 'revenue'])


################################################################################
# Financial metrics computation
################################################################################

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly financial metrics per team.

    The function performs the following steps:

    * Convert the ``date`` column to datetime and extract the monthly period.
    * Compute the burned value as ``hours × rate``.
    * Group by month and team: sum hours, burned value and revenue.
    * Calculate efficiency (revenue/burned value) and profit/loss (revenue - burned value).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns ['date','team','hours','rate','revenue'].

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['month','team','hours','burned_value','revenue','efficiency','profit_loss'].
    """
    df = df.copy()
    # Convert dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.to_period('M')
    # Compute burned value
    df['hours'] = pd.to_numeric(df['hours'], errors='coerce')
    df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df['burned_value'] = df['hours'] * df['rate']
    # Group by month and team
    agg = df.groupby(['month', 'team']).agg({
        'hours': 'sum',
        'burned_value': 'sum',
        'revenue': 'sum'
    }).reset_index()
    # Calculate efficiency and profit/loss
    agg['efficiency'] = np.where(agg['burned_value'] != 0,
                                 agg['revenue'] / agg['burned_value'], np.nan)
    agg['profit_loss'] = agg['revenue'] - agg['burned_value']
    return agg


################################################################################
# PDF generation utilities
################################################################################

class FinancialPDF(FPDF):
    """
    Custom FPDF subclass that includes page numbering in the footer.
    """
    def footer(self) -> None:
        # Position at 15 mm from bottom
        self.set_y(-15)
        # Use an italic font for footer; DejaVu if available
        try:
            self.set_font('DejaVu', 'I', 8)
        except Exception:
            self.set_font('Arial', 'I', 8)
        # Footer text
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def _sanitize_text(text: str) -> str:
    """
    Replace characters that are not supported by the default FPDF core fonts
    (latin-1 encoding). Converts curly quotes, long dashes and bullets to
    simpler equivalents. This ensures that even if the fallback font is used
    instead of a Unicode font, the text will not raise encoding errors.

    Parameters
    ----------
    text : str
        Input string to sanitise.

    Returns
    -------
    str
        Sanitised string safe for FPDF.
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
    company_name: str = "Report",
    summary: Optional[str] = None,
) -> None:
    """
    Generate a PDF report with the metrics and an optional executive summary.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        DataFrame containing columns ['month','team','hours','burned_value','revenue','efficiency','profit_loss'].
    output_path : str
        Path where the PDF will be saved.
    company_name : str, optional
        Name of the organisation or project to display in the report title.
    summary : str, optional
        Executive summary to include at the beginning of the report. If omitted
        or ``None``, only a brief introduction is shown.
    """
    pdf = FinancialPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Register Unicode-capable fonts (DejaVu). FPDF caches added fonts so repeated
    # calls are safe.
    try:
        pdf.add_font('DejaVu', '', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', uni=True)
        pdf.add_font('DejaVu', 'B', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', uni=True)
        pdf.add_font('DejaVu', 'I', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf', uni=True)
    except Exception:
        pass

    # Title
    try:
        pdf.set_font('DejaVu', 'B', 16)
    except Exception:
        pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f'FINANCIAL REPORT - {company_name}', ln=True)
    pdf.ln(4)
    # Introduction and summary
    try:
        pdf.set_font('DejaVu', '', 11)
    except Exception:
        pdf.set_font('Arial', '', 11)
    intro = (
        "This report presents the evolution of total hours worked, burned value, revenue and efficiency "
        "for each team by month. The values are calculated automatically from the uploaded Excel files."
    )
    pdf.multi_cell(0, 6, _sanitize_text(intro))
    pdf.ln(4)
    if summary:
        # Add header for the summary
        try:
            pdf.set_font('DejaVu', 'B', 12)
        except Exception:
            pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 7, "Executive summary", ln=True)
        try:
            pdf.set_font('DejaVu', '', 10)
        except Exception:
            pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, _sanitize_text(summary.strip()))
        pdf.ln(4)

    # Table headers
    try:
        pdf.set_font('DejaVu', 'B', 10)
    except Exception:
        pdf.set_font('Arial', 'B', 10)
    headers = ['Month', 'Team', 'Hours', 'Burned value', 'Revenue', 'Efficiency', 'Profit/Loss']
    col_widths = [25, 35, 25, 35, 30, 25, 40]
    for header, width in zip(headers, col_widths):
        pdf.cell(width, 8, header, 1, 0, 'C')
    pdf.ln()

    # Table rows
    try:
        pdf.set_font('DejaVu', '', 9)
    except Exception:
        pdf.set_font('Arial', '', 9)
    for _, row in df_metrics.iterrows():
        pdf.cell(col_widths[0], 7, _sanitize_text(str(row['month'])), 1)
        pdf.cell(col_widths[1], 7, _sanitize_text(str(row['team'])), 1)
        pdf.cell(col_widths[2], 7, f"{row['hours']:.1f}", 1, 0, 'R')
        pdf.cell(col_widths[3], 7, f"${row['burned_value']:.2f}", 1, 0, 'R')
        pdf.cell(col_widths[4], 7, f"${row['revenue']:.2f}", 1, 0, 'R')
        eff = row['efficiency'] * 100 if pd.notna(row['efficiency']) else 0
        pdf.cell(col_widths[5], 7, f"{eff:.1f}%", 1, 0, 'R')
        pdf.cell(col_widths[6], 7, f"${row['profit_loss']:.2f}", 1, 0, 'R')
        pdf.ln()

    # Save PDF
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf.output(output_path)
