"""
Convert docs/project_comprehensive_report.md into a high-quality PDF document.
Uses headless Edge/Chrome with modern CSS print engine and fallbacks to xhtml2pdf.
"""

import os
import re
import sys
import base64
import subprocess
import markdown

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

MD_PATH = os.path.join(REPO_ROOT, "docs", "project_comprehensive_report.md")
HTML_PATH = os.path.join(REPO_ROOT, "docs", "project_comprehensive_report.html")
PDF_PATH = os.path.join(REPO_ROOT, "docs", "project_comprehensive_report.pdf")


def convert_math_formulas(text: str) -> str:
    """
    Convert simple LaTeX math syntax into clean HTML strings for PDF rendering.
    """
    # Specific high-level formula replacements
    text = text.replace(
        r"\sum_{i \in \text{TP}(T)} \Big( \text{SaveRate} \times \text{PreservedValue}_i - \text{Cost}_{\text{offer}} \Big) - \sum_{j \in \text{FP}(T)} \text{Cost}_{\text{offer}}",
        "&Sigma;<sub>i &isin; TP(T)</sub> [ SaveRate &times; PreservedValue<sub>i</sub> - Cost<sub>offer</sub> ] - &Sigma;<sub>j &isin; FP(T)</sub> Cost<sub>offer</sub>",
    )
    text = text.replace(
        r"\text{Cost}(\text{False Negative}) \approx \$800 \text{ to } \$1,500 \quad \gg \quad \text{Cost}(\text{False Positive}) = \$35.00",
        "Cost(False Negative) &asymp; $800 to $1,500 &nbsp;&nbsp;&raquo;&nbsp;&nbsp; Cost(False Positive) = $35.00",
    )
    text = text.replace(
        r"P(\text{churn}) \ge 0.50",
        "P(churn) &ge; 0.50",
    )
    text = text.replace(
        r"T \in [0.05, 0.95]",
        "T &isin; [0.05, 0.95]",
    )

    # Clean general LaTeX symbols
    replacements = {
        r"\text{CLV}_i": "CLV<sub>i</sub>",
        r"\text{CLV}": "CLV",
        r"\text{MonthlyCharges}_i": "MonthlyCharges<sub>i</sub>",
        r"\text{MonthlyCharges}": "MonthlyCharges",
        r"\text{tenure}_i": "tenure<sub>i</sub>",
        r"\text{tenure}": "tenure",
        r"\text{Preserved Value}_i": "Preserved Value<sub>i</sub>",
        r"\text{PreservedValue}_i": "PreservedValue<sub>i</sub>",
        r"\text{Preserved Value}": "Preserved Value",
        r"\text{PreservedValue}": "PreservedValue",
        r"\text{Net Financial Savings}": "Net Financial Savings",
        r"\text{Cost}_{\text{offer}}": "Cost<sub>offer</sub>",
        r"\text{SaveRate}": "SaveRate",
        r"\text{TP}": "TP",
        r"\text{FP}": "FP",
        r"\text{FN}": "FN",
        r"\text{offer}": "offer",
        r"\text{churn}": "churn",
        r"\times": "&times;",
        r"\ge": "&ge;",
        r"\le": "&le;",
        r"\gg": "&raquo;",
        r"\chi^2": "&chi;&sup2;",
        r"\rho": "&rho;",
        r"\approx": "&asymp;",
        r"\in": "&isin;",
        r"\sum": "&Sigma;",
        r"\Big(": "[",
        r"\Big)": "]",
        r"\quad": "&nbsp;&nbsp;",
        r"\_": "_",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Replace block math
    text = re.sub(
        r"\$\$(.*?)\$\$",
        r'<div class="math-block"><em>\1</em></div>',
        text,
        flags=re.DOTALL,
    )
    # Replace inline math
    text = re.sub(
        r"\$(.*?)\$",
        r'<span class="math-inline"><em>\1</em></span>',
        text,
    )
    return text


def embed_local_images_as_base64(html_content: str, base_dir: str) -> str:
    """
    Find <img> tags referencing local paths and embed them as base64 data URIs
    so PDF renderers load all images without path resolution issues.
    """
    def replace_img(match):
        src = match.group(1)
        # Check relative paths
        candidate_paths = [
            os.path.normpath(os.path.join(base_dir, src)),
            os.path.normpath(os.path.join(REPO_ROOT, src.lstrip("../"))),
            os.path.normpath(os.path.join(REPO_ROOT, "reports", src.lstrip("../reports/"))),
        ]
        for p in candidate_paths:
            if os.path.exists(p) and not os.path.isdir(p):
                try:
                    ext = os.path.splitext(p)[1].lower().replace(".", "")
                    mime = f"image/{ext}" if ext in ["png", "jpg", "jpeg", "gif", "webp"] else "image/png"
                    with open(p, "rb") as img_f:
                        encoded = base64.b64encode(img_f.read()).decode("utf-8")
                    return f'src="data:{mime};base64,{encoded}"'
                except Exception as e:
                    print(f"Warning embedding {p}: {e}")
        return match.group(0)

    return re.sub(r'src="([^"]+)"', replace_img, html_content)


def convert_mermaid_diagrams(text: str) -> str:
    """
    Convert mermaid code blocks into clean HTML/CSS flowchart containers.
    """
    diagram1_html = """
    <div class="flowchart-container">
        <div class="fc-row">
            <span class="fc-card">1. Raw Data Ingestion</span> <span class="fc-arrow">&rarr;</span>
            <span class="fc-card">2. Data Quality & Quirk Fixes</span> <span class="fc-arrow">&rarr;</span>
            <span class="fc-card">3. Feature Engineering Pipeline</span> <span class="fc-arrow">&rarr;</span>
            <span class="fc-card">4. Baseline Model</span>
        </div>
        <div class="fc-row" style="margin-top: 10px;">
            <span class="fc-card">5. Stratified 5-Fold CV & Optuna</span> <span class="fc-arrow">&rarr;</span>
            <span class="fc-card">6. MLflow Tracking</span> <span class="fc-arrow">&rarr;</span>
            <span class="fc-card champion">7. Champion Tuned XGBoost</span>
        </div>
        <div class="fc-row" style="margin-top: 10px;">
            <span class="fc-card highlight">SHAP Explainability</span> &bull;
            <span class="fc-card highlight">Segment Error Audit</span> &bull;
            <span class="fc-card highlight">Threshold Optimization (T*=0.30)</span>
        </div>
    </div>
    """

    diagram2_html = """
    <div class="flowchart-container">
        <div class="fc-row">
            <span class="fc-card">Raw Telco DataFrame</span> <span class="fc-arrow">&rarr;</span>
            <span class="fc-card">clean_raw_data (Fix TotalCharges & Collapsing)</span> <span class="fc-arrow">&rarr;</span>
            <span class="fc-card">create_engineered_features</span>
        </div>
        <div class="fc-row" style="margin-top: 10px;">
            <span class="fc-badge">tenure_bucket</span>
            <span class="fc-badge">avg_monthly_spend</span>
            <span class="fc-badge">services_count</span>
            <span class="fc-badge">charge_ratio</span>
            <span class="fc-badge">clv</span>
            <span class="fc-badge">interaction flags</span>
        </div>
        <div class="fc-row" style="margin-top: 10px;">
            <span class="fc-arrow">&darr;</span>
        </div>
        <div class="fc-row">
            <span class="fc-card">ColumnTransformer (StandardScaler + OneHotEncoder)</span> <span class="fc-arrow">&rarr;</span>
            <span class="fc-card champion">34 Clean Standardized Features</span>
        </div>
    </div>
    """

    diagram3_html = """
    <div class="flowchart-container">
        <div class="fc-row">
            <span class="fc-card">Customer Risk Probability P</span> <span class="fc-arrow">&rarr;</span>
            <span class="fc-card highlight">Is P &ge; 0.30?</span>
        </div>
        <div class="fc-row" style="margin-top: 10px; display: flex;">
            <div style="flex: 1; text-align: center; border-right: 1px dashed #cbd5e0; padding: 6px;">
                <strong style="color: #2b6cb0;">YES (Target for Retention Outreach)</strong><br>
                <span class="fc-badge" style="background:#def7ec; color:#03543f;">True Positive: 45% Save Rate &times; 12mo Revenue - $35</span><br>
                <span class="fc-badge" style="background:#fde8e8; color:#9b1c1c;">False Positive: $35 Offer Cost Wasted</span>
            </div>
            <div style="flex: 1; text-align: center; padding: 6px;">
                <strong style="color: #718096;">NO (Standard Invoicing)</strong><br>
                <span class="fc-badge" style="background:#fde8e8; color:#9b1c1c;">False Negative: 100% Future Revenue Lost</span><br>
                <span class="fc-badge" style="background:#def7ec; color:#03543f;">True Negative: Zero Retention Cost</span>
            </div>
        </div>
    </div>
    """

    # Replace mermaid blocks
    text = re.sub(r"```mermaid\s+flowchart LR.*?```", diagram1_html, text, flags=re.DOTALL)
    text = re.sub(r"```mermaid\s+flowchart TD.*?```", diagram2_html, text, flags=re.DOTALL)
    text = re.sub(r"```mermaid\s+graph TD.*?```", diagram3_html, text, flags=re.DOTALL)
    return text


def generate_html_document(md_path: str, html_path: str):
    """
    Parse Markdown and assemble a styled HTML file optimized for PDF printing.
    """
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert diagrams and LaTeX formulas
    cleaned_md = convert_mermaid_diagrams(md_text)
    cleaned_md = convert_math_formulas(cleaned_md)

    # Convert to HTML with tables, fenced code, and TOC extensions
    extensions = ["tables", "fenced_code", "toc", "attr_list", "nl2br"]
    html_body = markdown.markdown(cleaned_md, extensions=extensions)

    # Embed local images as base64
    docs_dir = os.path.dirname(md_path)
    html_body = embed_local_images_as_base64(html_body, docs_dir)

    # Full styled HTML template
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Telco Customer Churn Prediction - Comprehensive Project Report</title>
<style>
    @page {{
        size: A4;
        margin: 18mm 16mm 18mm 16mm;
        @bottom-right {{
            content: "Page " counter(page);
            font-size: 9pt;
            color: #7f8c8d;
        }}
    }}
    
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 10.5pt;
        line-height: 1.55;
        color: #2c3e50;
        background-color: #ffffff;
        margin: 0;
        padding: 0;
    }}
    
    h1 {{
        color: #1a365d;
        font-size: 22pt;
        font-weight: 800;
        border-bottom: 2.5px solid #2b6cb0;
        padding-bottom: 8px;
        margin-top: 0;
        margin-bottom: 12px;
    }}
    
    h2 {{
        color: #2b6cb0;
        font-size: 15pt;
        font-weight: 700;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 5px;
        margin-top: 26px;
        margin-bottom: 12px;
        page-break-after: avoid;
    }}
    
    h3 {{
        color: #2d3748;
        font-size: 12.5pt;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 8px;
        page-break-after: avoid;
    }}
    
    h4 {{
        color: #4a5568;
        font-size: 11pt;
        font-weight: 600;
        margin-top: 14px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }}
    
    p {{
        margin-top: 0;
        margin-bottom: 10px;
    }}
    
    blockquote {{
        border-left: 4px solid #3182ce;
        background-color: #ebf8ff;
        color: #2b6cb0;
        padding: 10px 14px;
        margin: 12px 0;
        border-radius: 0 4px 4px 0;
        font-size: 10pt;
    }}
    
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0 18px 0;
        font-size: 9pt;
        page-break-inside: avoid;
    }}
    
    table th {{
        background-color: #2b6cb0;
        color: #ffffff;
        font-weight: 700;
        text-align: left;
        padding: 7px 9px;
        border: 1px solid #cbd5e0;
    }}
    
    table td {{
        padding: 6px 9px;
        border: 1px solid #e2e8f0;
        vertical-align: top;
    }}
    
    table tr:nth-child(even) {{
        background-color: #f7fafc;
    }}
    
    img {{
        max-width: 95%;
        height: auto;
        display: block;
        margin: 14px auto;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.06);
        page-break-inside: avoid;
    }}
    
    code {{
        font-family: Consolas, Monaco, "Courier New", monospace;
        font-size: 9pt;
        background-color: #edf2f7;
        color: #c53030;
        padding: 1px 4px;
        border-radius: 3px;
    }}
    
    pre {{
        background-color: #1a202c;
        color: #f7fafc;
        padding: 12px 14px;
        border-radius: 6px;
        font-size: 8.5pt;
        line-height: 1.45;
        overflow-x: auto;
        margin: 12px 0;
        page-break-inside: avoid;
    }}
    
    pre code {{
        background-color: transparent;
        color: #f7fafc;
        padding: 0;
    }}
    
    .math-block {{
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 8px 12px;
        margin: 10px 0;
        text-align: center;
        font-family: Georgia, serif;
    }}
    
    .math-inline {{
        font-family: Georgia, serif;
        font-style: italic;
    }}

    .flowchart-container {{
        background-color: #f8fafc;
        border: 1.5px solid #cbd5e0;
        border-radius: 8px;
        padding: 12px 14px;
        margin: 14px 0;
        font-size: 8.5pt;
        page-break-inside: avoid;
    }}

    .fc-row {{
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: 6px;
    }}

    .fc-card {{
        background: #ffffff;
        border: 1px solid #cbd5e0;
        padding: 5px 10px;
        border-radius: 4px;
        font-weight: 600;
        color: #2b6cb0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}

    .fc-card.champion {{
        background: #2b6cb0;
        color: #ffffff;
        border-color: #1a365d;
    }}

    .fc-card.highlight {{
        background: #ebf8ff;
        color: #2b6cb0;
        border-color: #bee3f8;
    }}

    .fc-arrow {{
        color: #718096;
        font-weight: bold;
        font-size: 11pt;
    }}

    .fc-badge {{
        display: inline-block;
        background: #edf2f7;
        color: #2d3748;
        padding: 3px 8px;
        border-radius: 4px;
        margin: 2px 4px;
        font-size: 8pt;
    }}
    
    hr {{
        border: 0;
        height: 1px;
        background: #e2e8f0;
        margin: 20px 0;
    }}
    
    ul, ol {{
        margin-top: 0;
        margin-bottom: 10px;
        padding-left: 22px;
    }}
    
    li {{
        margin-bottom: 4px;
    }}
    
    .page-break {{
        page-break-before: always;
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Generated intermediate HTML at {html_path}")
    return html_path


def render_pdf_with_browser(html_path: str, pdf_path: str) -> bool:
    """
    Attempt PDF conversion using Headless Microsoft Edge or Google Chrome.
    """
    browser_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]

    chosen_browser = None
    for b in browser_candidates:
        if os.path.exists(b):
            chosen_browser = b
            break

    if not chosen_browser:
        print("No headless Chrome or Edge executable found.")
        return False

    print(f"Rendering PDF via Headless Browser: {chosen_browser}")
    file_url = f"file:///{os.path.abspath(html_path).replace(os.sep, '/')}"

    cmd = [
        chosen_browser,
        "--headless=new",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--no-pdf-header-footer",
        f"--print-to-pdf={os.path.abspath(pdf_path)}",
        file_url,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
            print(f"Successfully generated PDF: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")
            return True
        else:
            print(f"Headless browser output: {proc.stdout} {proc.stderr}")
    except Exception as e:
        print(f"Headless browser execution error: {e}")

    return False


def render_pdf_with_xhtml2pdf(html_path: str, pdf_path: str) -> bool:
    """
    Fallback PDF conversion using xhtml2pdf.
    """
    from xhtml2pdf import pisa
    print("Rendering PDF via pure-Python xhtml2pdf...")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        with open(pdf_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)

        if not pisa_status.err and os.path.exists(pdf_path):
            print(f"Successfully generated PDF via xhtml2pdf: {pdf_path}")
            return True
        else:
            print(f"xhtml2pdf reported errors: {pisa_status.err}")
    except Exception as e:
        print(f"xhtml2pdf error: {e}")
    return False


def generate_pdf():
    print("--- 1. Generating Styled HTML from Markdown ---")
    html_path = generate_html_document(MD_PATH, HTML_PATH)

    print("--- 2. Converting HTML to PDF ---")
    success = render_pdf_with_browser(html_path, PDF_PATH)
    if not success:
        success = render_pdf_with_xhtml2pdf(html_path, PDF_PATH)

    if success:
        print(f"\nSUCCESS: PDF Generated at: {PDF_PATH}")
    else:
        print("\nERROR: Failed to generate PDF.")


if __name__ == "__main__":
    generate_pdf()
