"""
workflows/dashboard_style.py — AT&T-aligned CSS for the Streamlit dashboard.

A single source of truth for the dashboard's look. The brand tokens are pulled
straight from :data:`DSSP2026.core.style.ATT_COLORS`, so the app's chrome stays
in lockstep with the matplotlib/seaborn figures it renders — change a hex in
``core/style.py`` and both the plots and this CSS move together.

The visual language follows the annotated TeleLogs explorer page: a clean white
background with faint cool-gray panels, **Inter** for body copy and **JetBrains
Mono** for the small uppercase, wide-tracked labels, AT&T blue accents, a navy
table header band, blue hover tints, soft 6px-radius cards, and a sticky title
rule. Nothing here changes behaviour — it is presentation only.

Usage (in the dashboard's ``render``)::

    import streamlit as st
    from DSSP2026.workflows.dashboard_style import inject_dashboard_css
    inject_dashboard_css(st)        # once, right after set_page_config
"""

from __future__ import annotations

try:
    from DSSP2026.core.style import ATT_COLORS
except Exception:  # pragma: no cover - defensive; mirror the dashboard fallback
    ATT_COLORS = {
        "att_blue": "#00A8E0", "navy": "#002A5C", "deep_blue": "#0057B8",
        "sky": "#7FD3EF", "pale_sky": "#D6F0FA", "gray_900": "#1A1A1A",
        "gray_700": "#333333", "gray_500": "#666666", "gray_300": "#BBBBBB",
        "gray_100": "#F2F2F2", "white": "#FFFFFF", "green": "#2E7031",
        "magenta": "#C8102E", "orange": "#E55A0B", "teal": "#00857C",
        "gold": "#C99000",
    }


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha})"


# Web fonts matching the reference page (Inter + JetBrains Mono).
_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=JetBrains+Mono:wght@400;500;700&"
    "family=Inter:wght@400;500;600;700;800;900&display=swap');"
)


def dashboard_css() -> str:
    """Return the full ``<style>`` block (brand tokens baked in from ATT_COLORS).

    Pure string builder so it is unit-testable without Streamlit. Inject it with
    :func:`inject_dashboard_css`, or pass the string to ``st.markdown(...,
    unsafe_allow_html=True)`` yourself.
    """
    c = ATT_COLORS
    # Derived surfaces: faint cool grays tuned toward the brand blue, mirroring
    # the reference page's --panel / --panel-2 without inventing new base hexes.
    panel = c["gray_100"]                 # sidebar / panel background
    panel_2 = "#EEF3F8"                   # tables / chips — one step cooler/deeper
    rule = c["gray_300"]                  # borders / dividers
    ink = c["gray_900"]                   # body text
    ink_dim = c["gray_500"]               # secondary text
    accent = c["att_blue"]                # primary accent (brand mark)
    accent_strong = c["deep_blue"]        # buttons / stronger accent
    navy = c["navy"]                      # headers / table header band
    hover_tint = _rgba(c["att_blue"], 0.08)
    focus_ring = _rgba(c["att_blue"], 0.18)
    accent_soft = _rgba(c["att_blue"], 0.12)

    mono = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"
    sans = ("'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', "
            "Helvetica, Arial, sans-serif")

    return f"""<style>
  {_FONT_IMPORT}

  :root {{
    --att-blue:   {accent};
    --att-navy:   {navy};
    --att-deep:   {accent_strong};
    --ink:        {ink};
    --ink-dim:    {ink_dim};
    --rule:       {rule};
    --panel:      {panel};
    --panel-2:    {panel_2};
    --hover-tint: {hover_tint};
    --focus-ring: {focus_ring};
    --accent-soft:{accent_soft};
  }}

  /* ---- Base surfaces & body type --------------------------------------- */
  .stApp {{
    background: {c['white']};
    color: var(--ink);
    font-family: {sans};
  }}
  .block-container {{
    padding-top: 2.4rem;
    max-width: 1200px;
  }}

  /* ---- Headings: navy, tight, with a thin brand rule ------------------- */
  h1, .stApp h1 {{
    font-family: {sans};
    font-weight: 800;
    letter-spacing: -0.025em;
    color: var(--att-navy);
    line-height: 1.08;
  }}
  h2, h3, .stApp h2, .stApp h3 {{
    font-family: {sans};
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--att-navy);
  }}
  /* Section subheaders (st.subheader renders an h3) get a hairline divider,
     echoing the reference page's mono section labels. */
  .stApp h3 {{
    padding-bottom: 6px;
    border-bottom: 1px solid var(--rule);
    margin-top: 1.5rem;
  }}

  /* The dashboard's own brand rule under the title. */
  .att-rule {{
    height: 4px; border: 0;
    background: var(--att-blue);
    margin: 0 0 1.25rem 0;
    border-radius: 2px;
  }}

  /* ---- Captions & small labels: mono, uppercase, wide tracking --------- */
  .stCaption, [data-testid="stCaptionContainer"] {{
    font-family: {mono};
    color: var(--ink-dim);
    font-size: 12px;
    letter-spacing: 0.04em;
  }}

  /* ---- Sidebar: faint cool panel with a brand left edge ---------------- */
  [data-testid="stSidebar"] {{
    background: var(--panel);
    border-right: 1px solid var(--rule);
  }}
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {{
    font-family: {mono};
    font-size: 12px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-weight: 500;
    color: var(--ink-dim);
    border-bottom: 1px solid var(--rule);
    padding-bottom: 8px;
    margin-top: 1.4rem;
  }}

  /* ---- Metrics: card-like, brand value color --------------------------- */
  [data-testid="stMetric"] {{
    background: var(--panel-2);
    border: 1px solid var(--rule);
    border-radius: 6px;
    padding: 14px 16px;
  }}
  [data-testid="stMetricLabel"] {{
    font-family: {mono};
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-dim);
  }}
  [data-testid="stMetricValue"] {{
    color: var(--att-deep);
    font-weight: 800;
    font-variant-numeric: tabular-nums;
  }}

  /* ---- Buttons: AT&T blue, mono caps, soft lift ------------------------ */
  .stButton > button, .stDownloadButton > button {{
    font-family: {mono};
    font-size: 12px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 700;
    color: {c['white']};
    background: var(--att-deep);
    border: 1px solid var(--att-deep);
    border-radius: 6px;
    padding: 8px 18px;
    transition: background 0.15s, box-shadow 0.15s, transform 0.1s;
  }}
  .stButton > button:hover, .stDownloadButton > button:hover {{
    background: var(--att-blue);
    border-color: var(--att-blue);
    box-shadow: 0 2px 10px var(--accent-soft);
    transform: translateY(-1px);
  }}
  .stButton > button:focus, .stDownloadButton > button:focus {{
    outline: none;
    box-shadow: 0 0 0 3px var(--focus-ring);
  }}

  /* ---- Inputs / selects: brand focus ring ------------------------------ */
  [data-testid="stSidebar"] input,
  [data-baseweb="select"] > div,
  .stTextInput input, .stNumberInput input {{
    border-radius: 6px !important;
  }}
  [data-baseweb="select"] > div:focus-within,
  .stTextInput input:focus, .stNumberInput input:focus {{
    border-color: var(--att-blue) !important;
    box-shadow: 0 0 0 3px var(--focus-ring) !important;
  }}

  /* ---- Radios / tabs accent -------------------------------------------- */
  .stRadio [aria-checked="true"] {{
    border-color: var(--att-blue) !important;
  }}

  /* ---- Tables / dataframes: navy header band, blue row hover ----------- */
  [data-testid="stDataFrame"] {{
    border: 1px solid var(--rule);
    border-radius: 6px;
    overflow: hidden;
  }}
  [data-testid="stTable"] table, [data-testid="stDataFrame"] table {{
    border-collapse: collapse;
    width: 100%;
  }}
  [data-testid="stTable"] thead th {{
    background: var(--att-navy);
    color: {c['white']};
    font-family: {mono};
    font-size: 11px;
    letter-spacing: 0.05em;
    font-weight: 500;
    text-align: left;
    padding: 9px 13px;
  }}
  [data-testid="stTable"] tbody td {{
    font-variant-numeric: tabular-nums;
    padding: 7px 13px;
    border-bottom: 1px solid var(--rule);
  }}
  [data-testid="stTable"] tbody tr:hover td {{
    background: var(--hover-tint);
  }}

  /* ---- Expander: panel card -------------------------------------------- */
  [data-testid="stExpander"] {{
    border: 1px solid var(--rule);
    border-radius: 6px;
    background: var(--panel-2);
  }}
  [data-testid="stExpander"] summary {{
    font-family: {mono};
    font-size: 12px;
    letter-spacing: 0.05em;
    color: var(--att-navy);
    font-weight: 700;
  }}

  /* ---- Alerts: tint with brand blue left edge -------------------------- */
  [data-testid="stAlert"] {{
    border-radius: 6px;
    border-left: 3px solid var(--att-blue);
  }}

  /* ---- Dividers -------------------------------------------------------- */
  hr {{ border-color: var(--rule); }}

  /* ---- Subtle entrance, matching the reference page's fade-in ---------- */
  .block-container > div {{ animation: dsspFade 0.2s ease; }}
  @keyframes dsspFade {{
    from {{ opacity: 0; transform: translateY(4px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}
</style>"""


def inject_dashboard_css(st) -> None:
    """Inject the dashboard CSS into a Streamlit app (call once after
    ``set_page_config``). Kept as a one-liner so the dashboard's ``render`` stays
    uncluttered and the styling lives entirely in this module.
    """
    st.markdown(dashboard_css(), unsafe_allow_html=True)