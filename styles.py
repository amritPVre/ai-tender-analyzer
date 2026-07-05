"""BAESS.APP brand styling for Streamlit."""

from config import COLORS


def inject_custom_css() -> str:
    """Return CSS block for st.markdown unsafe_allow_html."""
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background-color: #F5F7FA;
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS["navy"]} 0%, #152a4a 100%);
    }}

    [data-testid="stSidebar"] * {{
        color: #E8EDF5 !important;
    }}

    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label {{
        color: #E8EDF5 !important;
    }}

    .main-header {{
        background: {COLORS["navy"]};
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 5px solid {COLORS["teal"]};
    }}

    .main-header h1 {{
        color: white !important;
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
    }}

    .main-header p {{
        color: {COLORS["slate"]} !important;
        margin: 0.35rem 0 0 0;
        font-size: 0.95rem;
    }}

    .section-card {{
        background: {COLORS["cream"]};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #E8DFC8;
        box-shadow: 0 2px 8px rgba(11, 31, 58, 0.06);
    }}

    .section-title {{
        color: {COLORS["navy"]};
        font-weight: 700;
        font-size: 1.1rem;
        border-bottom: 2px solid {COLORS["teal"]};
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }}

    .metric-card {{
        background: white;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid {COLORS["teal"]};
        box-shadow: 0 1px 4px rgba(11, 31, 58, 0.08);
        min-height: 90px;
    }}

    .metric-card.urgent {{
        border-left-color: {COLORS["amber"]};
        background: #FFF8F0;
    }}

    .urgent-badge {{
        background: {COLORS["amber"]};
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-left: 6px;
    }}

    .upload-prompt {{
        text-align: center;
        padding: 3rem 2rem;
        background: white;
        border-radius: 16px;
        border: 2px dashed {COLORS["teal"]};
        margin: 2rem 0;
    }}

    .upload-prompt h2 {{
        color: {COLORS["navy"]};
    }}

    .upload-prompt p {{
        color: {COLORS["slate"]};
    }}

    .recommendation-green {{
        background: #D1FAE5;
        border: 1px solid #10B981;
        border-radius: 10px;
        padding: 1rem;
        color: #065F46;
        font-weight: 600;
    }}

    .recommendation-amber {{
        background: #FEF3C7;
        border: 1px solid {COLORS["amber"]};
        border-radius: 10px;
        padding: 1rem;
        color: #92400E;
        font-weight: 600;
    }}

    .recommendation-red {{
        background: #FEE2E2;
        border: 1px solid #EF4444;
        border-radius: 10px;
        padding: 1rem;
        color: #991B1B;
        font-weight: 600;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: {COLORS["navy"]};
        border-radius: 10px;
        padding: 6px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        color: #A8B4C8 !important;
        border-radius: 8px;
        font-weight: 500;
    }}

    .stTabs [aria-selected="true"] {{
        background: {COLORS["teal"]} !important;
        color: white !important;
    }}

    div[data-testid="stMetric"] {{
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid {COLORS["teal"]};
        box-shadow: 0 1px 4px rgba(11, 31, 58, 0.08);
    }}

    .stAlert {{
        border-radius: 10px;
    }}

    footer {{
        visibility: hidden;
    }}

    .baess-watermark {{
        color: {COLORS["slate"]};
        font-size: 0.75rem;
        text-align: center;
        margin-top: 2rem;
    }}
    </style>
    """
