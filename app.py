# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from sklearn.inspection import permutation_importance

from src.data import (
    load_raw_data,
    clean_data,
    split_features_target,
)

from src.preprocessing import build_preprocessor
from src.models import available_models

from src.evaluation import (
    train_and_evaluate,
    compare_models,
    confusion_matrix_dataframe,
)

from src.interpretation import explain_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BankLens | Marketing Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

PAGE_BG = "#F3F6FC"
SIDEBAR_BG = "#F8FAFE"

CARD = "#FFFFFF"
CARD_SOFT = "#F8FAFF"
CARD_BLUE = "#F3F7FF"

BORDER = "#E1E7F0"
BORDER_STRONG = "#D2DBEA"

TEXT = "#20283A"
TEXT_2 = "#4C5870"
MUTED = "#8792A8"

BLUE = "#4A8EF6"
BLUE_DARK = "#3276DA"

CYAN = "#2EC7D3"
PURPLE = "#8B63E8"
PINK = "#ED5B9B"

GREEN = "#25B981"
AMBER = "#F1A62A"
RED = "#EA5B69"


# ============================================================
# ICONS
# ============================================================

ICON_PATHS = {
    "users": """
        <circle cx="9" cy="8" r="3"/>
        <path d="M3 20c0-4 2.5-6 6-6s6 2 6 6"/>
        <circle cx="17" cy="9" r="2.5"/>
        <path d="M15 15c3.7-.6 6 1.5 6 5"/>
    """,

    "percent": """
        <circle cx="7" cy="7" r="2"/>
        <circle cx="17" cy="17" r="2"/>
        <path d="M19 5L5 19"/>
    """,

    "database": """
        <ellipse cx="12" cy="5" rx="7" ry="3"/>
        <path d="M5 5v6c0 1.7 3.1 3 7 3s7-1.3 7-3V5"/>
        <path d="M5 11v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6"/>
    """,

    "phone": """
        <path d="M6 3h4l2 5-3 2c1.8 3.7 3.4 5.3 7 7l2-3 4 2v4c0 1-1 2-2 2C10 21 3 14 3 6c0-2 1-3 3-3z"/>
    """,

    "calendar": """
        <rect x="3" y="5" width="18" height="16" rx="2"/>
        <path d="M8 3v4M16 3v4M3 10h18"/>
    """,

    "trend": """
        <path d="M3 17l6-6 4 4 8-9"/>
        <path d="M16 6h5v5"/>
    """,
}


def icon_svg(
    name,
    size=20,
    color="currentColor",
    width=1.8,
):
    return (
        f"<svg "
        f"width='{size}' "
        f"height='{size}' "
        f"viewBox='0 0 24 24' "
        f"fill='none' "
        f"stroke='{color}' "
        f"stroke-width='{width}' "
        f"stroke-linecap='round' "
        f"stroke-linejoin='round'>"
        f"{ICON_PATHS[name]}"
        f"</svg>"
    )


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    f"""
<style>

* {{
    box-sizing: border-box;
}}


/* =========================================================
   GLOBAL
========================================================= */

html,
body,
[data-testid="stAppViewContainer"],
.stApp {{

    background:

        radial-gradient(
            circle at 70% -10%,
            rgba(90,132,255,.08),
            transparent 30%
        ),

        linear-gradient(
            135deg,
            #F6F8FD 0%,
            #F0F5FC 55%,
            #F7F4FD 100%
        );

    color: {TEXT};

    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}}


/* =========================================================
   STREAMLIT MENU
========================================================= */

#MainMenu {{
    visibility: visible !important;
}}

[data-testid="stToolbar"] {{
    visibility: visible !important;
    opacity: 1 !important;
}}

[data-testid="stHeader"] {{
    background: transparent !important;
    visibility: visible !important;
    z-index: 999999 !important;
}}

footer {{
    visibility: hidden;
}}


/* =========================================================
   PAGE WIDTH
========================================================= */

.block-container {{
    max-width: 1540px;

    padding-top: 1.1rem;
    padding-bottom: 4rem;

    padding-left: 2rem;
    padding-right: 2rem;
}}


/* =========================================================
   SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {{
    background:

        linear-gradient(
            180deg,
            #FAFBFE 0%,
            #F1F5FB 100%
        );

    border-right:
        1px solid {BORDER};
}}

section[data-testid="stSidebar"] > div {{
    background: transparent;
}}


.sidebar-brand {{
    display: flex;

    align-items: center;

    gap: 13px;

    margin-top: 7px;
    margin-bottom: 32px;
}}


.brand-logo {{
    width: 48px;
    height: 48px;

    border-radius: 12px;

    display: flex;

    align-items: center;
    justify-content: center;

    background:

        linear-gradient(
            145deg,
            #36A8F4,
            #6F6EE8 55%,
            #E55EB2
        );

    box-shadow:

        0 9px 24px
        rgba(90,105,210,.20);
}}


.logo-bars {{
    display: flex;

    align-items: flex-end;

    gap: 3px;

    height: 24px;
}}


.logo-bars span {{
    width: 4px;

    border-radius: 5px;

    background: white;
}}


.logo-bars span:nth-child(1) {{
    height: 9px;
}}

.logo-bars span:nth-child(2) {{
    height: 16px;
}}

.logo-bars span:nth-child(3) {{
    height: 22px;
}}

.logo-bars span:nth-child(4) {{
    height: 13px;
}}


.brand-name {{
    color: {TEXT};

    font-size: 21px;

    font-weight: 800;

    letter-spacing: -.35px;
}}


.brand-caption {{
    color: {MUTED};

    font-size: 10px;

    margin-top: 3px;
}}


.sidebar-label {{
    color: #9AA4B5;

    font-size: 9px;

    font-weight: 800;

    letter-spacing: 1.4px;

    text-transform: uppercase;

    margin-bottom: 10px;
}}


.sidebar-card {{
    background: white;

    padding: 15px;

    border-radius: 12px;

    border:
        1px solid {BORDER};

    box-shadow:

        0 5px 18px
        rgba(65,77,110,.055);

    margin-bottom: 12px;
}}


.sidebar-card-title {{
    color: {TEXT};

    font-size: 12px;

    font-weight: 700;
}}


.sidebar-card-text {{
    color: {MUTED};

    font-size: 10px;

    line-height: 1.55;

    margin-top: 5px;
}}


/* =========================================================
   HEADER
========================================================= */

.page-title {{
    color: {TEXT};

    font-size: 31px;

    font-weight: 820;

    line-height: 1.08;

    letter-spacing: -.65px;
}}


.page-subtitle {{
    color: {MUTED};

    font-size: 13px;

    margin-top: 6px;
}}


.header-chip {{
    display: inline-flex;

    align-items: center;

    gap: 8px;

    height: 39px;

    padding:
        0 13px;

    background: white;

    border:
        1px solid {BORDER_STRONG};

    border-radius: 9px;

    color: {TEXT_2};

    font-size: 11px;

    box-shadow:

        0 4px 14px
        rgba(61,76,111,.06);
}}


/* =========================================================
   SEARCH
========================================================= */

div[data-testid="stTextInput"] input {{
    height: 45px;

    background:
        rgba(255,255,255,.96) !important;

    color:
        {TEXT} !important;

    border:
        1px solid {BORDER_STRONG} !important;

    border-radius:
        10px !important;

    box-shadow:

        0 4px 15px
        rgba(53,73,109,.045);
}}


div[data-testid="stTextInput"] input:focus {{
    border-color:
        rgba(74,142,246,.65) !important;

    box-shadow:

        0 0 0 3px
        rgba(74,142,246,.10);
}}


/* =========================================================
   KPI CARDS
========================================================= */

.kpi-card {{
    position: relative;

    min-height: 125px;

    padding:
        17px 18px;

    overflow: hidden;

    background:

        linear-gradient(
            145deg,
            #FFFFFF 20%,
            var(--soft) 100%
        );

    border:
        1px solid var(--borderColor);

    border-radius:
        14px;

    box-shadow:

        0 7px 22px
        rgba(57,74,111,.075);

    transition:
        transform .18s ease,
        box-shadow .18s ease;

    animation:
        kpiPop
        .65s
        cubic-bezier(.16,1,.3,1)
        both;
}}


.kpi-card:hover {{
    transform:
        translateY(-3px);

    box-shadow:

        0 12px 30px
        rgba(57,74,111,.13);
}}


.kpi-card::before {{
    content: "";

    position: absolute;

    top: 0;
    left: 0;

    width: 100%;
    height: 3px;

    background:

        linear-gradient(
            90deg,
            transparent,
            var(--accent),
            transparent
        );

    opacity: .85;
}}


.kpi-top {{
    display: flex;

    align-items: center;

    justify-content: space-between;
}}


.kpi-title {{
    color: {TEXT_2};

    font-size: 11px;
}}


.kpi-icon {{
    width: 37px;
    height: 37px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 10px;

    background:
        var(--iconBg);

    box-shadow:

        0 5px 15px
        var(--shadowColor);
}}


.kpi-middle {{
    display: flex;

    justify-content: space-between;

    align-items: flex-end;

    margin-top: 8px;
}}


.kpi-value {{
    color: {TEXT};

    font-size: 28px;

    font-weight: 820;

    line-height: 1;
}}


.kpi-description {{
    color:
        var(--accent);

    font-size: 10px;

    font-weight: 700;

    margin-top: 11px;
}}


.sparkline {{
    width: 86px;

    height: 36px;

    overflow: visible;
}}


/* =========================================================
   NAVIGATION TABS
========================================================= */

.stTabs [data-baseweb="tab-list"] {{
    gap: 24px;

    border-bottom:
        1px solid #DCE3EE;

    margin-top: 9px;

    margin-bottom: 21px;
}}


.stTabs [data-baseweb="tab"] {{
    height: 47px;

    background:
        transparent !important;

    border:
        none !important;

    color:
        #657087 !important;

    padding:
        8px 4px 12px !important;

    font-size:
        13px !important;

    font-weight:
        540 !important;
}}


.stTabs [data-baseweb="tab"]:hover {{
    color:
        {TEXT} !important;
}}


.stTabs [aria-selected="true"] {{
    color:
        {BLUE_DARK} !important;

    font-weight:
        700 !important;
}}


.stTabs [data-baseweb="tab-highlight"] {{
    height:
        3px !important;

    border-radius:
        4px 4px 0 0;

    background:

        linear-gradient(
            90deg,
            {BLUE},
            {CYAN}
        ) !important;

    box-shadow:

        0 2px 8px
        rgba(74,142,246,.24);
}}


/* =========================================================
   CONTAINERS
========================================================= */

div[data-testid="stVerticalBlockBorderWrapper"] {{
    background:
        rgba(255,255,255,.97);

    border:
        1px solid {BORDER} !important;

    border-radius:
        14px;

    overflow: hidden;

    box-shadow:

        0 8px 24px
        rgba(61,77,110,.07);

    animation:
        cardReveal
        .55s
        ease-out
        both;
}}


/* =========================================================
   SECTION TEXT
========================================================= */

.section-label {{
    color: {BLUE};

    font-size: 9px;

    font-weight: 800;

    letter-spacing: 1.4px;

    text-transform: uppercase;

    margin-bottom: 5px;
}}


.section-title {{
    color: {TEXT};

    font-size: 18px;

    font-weight: 750;

    margin-bottom: 3px;
}}


.section-description {{
    color: {MUTED};

    font-size: 11px;

    margin-bottom: 14px;
}}


/* =========================================================
   STATUS ROWS
========================================================= */

.status-row {{
    min-height: 37px;

    display: flex;

    justify-content: space-between;

    align-items: center;

    padding:
        0 12px;

    margin-bottom: 7px;

    background:
        #F9FBFE;

    border:
        1px solid #E3EAF3;

    border-radius:
        8px;

    font-size: 10px;
}}


.status-name {{
    display: flex;

    align-items: center;

    gap: 8px;

    color: {TEXT_2};
}}


.status-dot {{
    width: 8px;
    height: 8px;

    border-radius: 50%;

    background:
        var(--status);

    box-shadow:
        0 0 8px
        var(--status);
}}


.status-value {{
    color:
        var(--status);

    font-weight: 700;
}}


/* =========================================================
   INPUTS
========================================================= */

div[data-baseweb="select"] > div,
div[data-baseweb="input"] {{
    background:
        white !important;

    border-color:
        {BORDER_STRONG} !important;

    color:
        {TEXT} !important;
}}


input {{
    color:
        {TEXT} !important;
}}


/* =========================================================
   BUTTONS
========================================================= */

.stButton > button,
div[data-testid="stFormSubmitButton"] button {{
    min-height: 41px;

    padding:
        0 18px;

    color:
        white;

    background:

        linear-gradient(
            90deg,
            #4A8EF6,
            #66A9F7
        );

    border: none;

    border-radius:
        9px;

    font-weight:
        650;

    box-shadow:

        0 7px 18px
        rgba(74,142,246,.18);
}}


.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {{
    transform:
        translateY(-1px);

    box-shadow:

        0 9px 22px
        rgba(74,142,246,.27);
}}


/* =========================================================
   METRICS
========================================================= */

div[data-testid="stMetric"] {{
    background:
        #FAFBFD;

    border:
        1px solid #E1E7F0;

    border-radius:
        10px;

    padding:
        13px 14px;
}}


div[data-testid="stMetricLabel"] {{
    color:
        {MUTED};
}}


div[data-testid="stMetricValue"] {{
    color:
        {TEXT};

    font-weight:
        790;
}}


/* =========================================================
   TABLE
========================================================= */

div[data-testid="stDataFrame"] {{
    border:
        1px solid {BORDER};

    border-radius:
        10px;

    overflow: hidden;
}}


details {{
    background:
        white;

    border:
        1px solid {BORDER} !important;

    border-radius:
        10px !important;
}}


p {{
    color:
        {TEXT_2};
}}


/* =========================================================
   ANIMATION
========================================================= */

@keyframes chartReveal {{

    0% {{
        opacity: 0;

        transform:
            translateY(14px)
            scale(.985);
    }}

    100% {{
        opacity: 1;

        transform:
            translateY(0)
            scale(1);
    }}
}}


@keyframes cardReveal {{

    0% {{
        opacity: 0;

        transform:
            translateY(10px);
    }}

    100% {{
        opacity: 1;

        transform:
            translateY(0);
    }}
}}


@keyframes kpiPop {{

    0% {{
        opacity: 0;

        transform:
            scale(.96);
    }}

    70% {{
        transform:
            scale(1.012);
    }}

    100% {{
        opacity: 1;

        transform:
            scale(1);
    }}
}}


div[data-testid="stPlotlyChart"] {{
    animation:
        chartReveal
        .70s
        cubic-bezier(.16,1,.3,1)
        both;
}}


iframe[title="streamlit.components.v1.html"] {{
    animation:
        chartReveal
        .70s
        cubic-bezier(.16,1,.3,1)
        both;

    border:
        none !important;
}}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# UTILITIES
# ============================================================

def hex_to_rgba(
    color,
    opacity,
):
    color = color.lstrip("#")

    r = int(
        color[0:2],
        16,
    )

    g = int(
        color[2:4],
        16,
    )

    b = int(
        color[4:6],
        16,
    )

    return (
        f"rgba({r},{g},{b},{opacity})"
    )


def section_header(
    label,
    title,
    description=None,
):
    html = (
        f"<div class='section-label'>"
        f"{label}"
        f"</div>"

        f"<div class='section-title'>"
        f"{title}"
        f"</div>"
    )

    if description:
        html += (
            f"<div class='section-description'>"
            f"{description}"
            f"</div>"
        )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


# ============================================================
# FEATURE NAME CLEANER
# ============================================================

def clean_feature_name(
    name,
):
    name = str(
        name
    )

    prefixes = [
        "categorical__",
        "numeric__",
        "category__",
        "number__",
        "cat__",
        "num__",
    ]

    for prefix in prefixes:

        if name.startswith(
            prefix
        ):
            name = name[
                len(prefix):
            ]

    return name


# ============================================================
# KPI HELPERS
# ============================================================

def sparkline_svg(
    values,
    color,
):
    width = 86
    height = 36

    minimum = min(
        values
    )

    maximum = max(
        values
    )

    spread = (
        maximum - minimum
        if maximum != minimum
        else 1
    )

    points = []

    for index, value in enumerate(
        values
    ):
        x = (
            index
            / max(
                len(values) - 1,
                1,
            )
            * width
        )

        y = (
            height
            - (
                (
                    value - minimum
                )
                / spread
                * 27
            )
            - 4
        )

        points.append(
            f"{x:.1f},{y:.1f}"
        )

    point_string = (
        " ".join(
            points
        )
    )

    filter_id = (
        "shadow"
        + color.replace(
            "#",
            "",
        )
    )

    return (
        f"<svg "
        f"class='sparkline' "
        f"viewBox='0 0 {width} {height}'>"

        "<defs>"

        f"<filter id='{filter_id}'>"

        "<feGaussianBlur "
        "stdDeviation='1.6' "
        "result='blur'/>"

        "<feMerge>"

        "<feMergeNode in='blur'/>"

        "<feMergeNode in='SourceGraphic'/>"

        "</feMerge>"

        "</filter>"

        "</defs>"

        f"<polyline "
        f"points='{point_string}' "
        f"fill='none' "
        f"stroke='{color}' "
        f"stroke-width='2.4' "
        f"stroke-linecap='round' "
        f"stroke-linejoin='round' "
        f"filter='url(#{filter_id})'/>"

        "</svg>"
    )


def kpi_card(
    title,
    value,
    description,
    accent,
    soft,
    icon_bg,
    icon,
    spark_values,
):
    sparkline = sparkline_svg(
        spark_values,
        accent,
    )

    html = (
        f"<div "
        f"class='kpi-card' "
        f"style='"

        f"--accent:{accent};"

        f"--soft:{soft};"

        f"--borderColor:"
        f"{hex_to_rgba(accent,.42)};"

        f"--iconBg:{icon_bg};"

        f"--shadowColor:"
        f"{hex_to_rgba(accent,.15)};"

        f"'>"

        "<div class='kpi-top'>"

        f"<div class='kpi-title'>"
        f"{title}"
        f"</div>"

        f"<div class='kpi-icon'>"
        f"{icon}"
        f"</div>"

        "</div>"

        "<div class='kpi-middle'>"

        f"<div class='kpi-value'>"
        f"{value}"
        f"</div>"

        f"{sparkline}"

        "</div>"

        f"<div class='kpi-description'>"
        f"{description}"
        f"</div>"

        "</div>"
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


# ============================================================
# STATUS ROW
# ============================================================

def status_row(
    name,
    value,
    color,
):
    st.markdown(
        (
            f"<div "
            f"class='status-row' "
            f"style='--status:{color};'>"

            "<div class='status-name'>"

            "<span class='status-dot'></span>"

            f"{name}"

            "</div>"

            f"<div class='status-value'>"
            f"{value}"
            f"</div>"

            "</div>"
        ),
        unsafe_allow_html=True,
    )


# ============================================================
# PLOTLY CONFIG
# ============================================================

PLOT_CONFIG = {
    "displayModeBar": "hover",

    "scrollZoom": True,

    "displaylogo": False,

    "responsive": True,

    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
    ],
}


# ============================================================
# CHART STYLE
# ============================================================

def style_chart(
    fig,
    height=390,
    legend=True,
    small=False,
):
    fig.update_layout(
        height=height,

        paper_bgcolor=
            "rgba(0,0,0,0)",

        plot_bgcolor=
            "#FCFDFF",

        font=dict(
            family=(
                "Inter, Segoe UI, Arial"
            ),

            size=(
                10
                if small
                else 11
            ),

            color=TEXT_2,
        ),

        margin=dict(
            l=(
                35
                if small
                else 45
            ),

            r=18,

            t=25,

            b=40,
        ),

        hoverlabel=dict(
            bgcolor="#FFFFFF",

            bordercolor="#CCD8E8",

            font_color=TEXT,

            font_size=11,
        ),

        legend=dict(
            orientation="h",

            x=1,

            xanchor="right",

            y=1.10,

            bgcolor=
                "rgba(255,255,255,0)",

            font=dict(
                color=TEXT_2,

                size=(
                    9
                    if small
                    else 10
                ),
            ),
        ),

        showlegend=legend,
    )

    fig.update_xaxes(
        showgrid=small,

        gridcolor=
            "rgba(190,203,222,.35)",

        griddash="dot",

        zeroline=False,

        linecolor="#D6DEEA",

        tickfont=dict(
            color="#7C8DA8",
        ),

        title_font=dict(
            color=TEXT_2,
        ),

        ticks="",
    )

    fig.update_yaxes(
        showgrid=True,

        gridcolor=
            "rgba(190,203,222,.44)",

        griddash="dot",

        zeroline=False,

        linecolor="#D6DEEA",

        tickfont=dict(
            color="#7C8DA8",
        ),

        title_font=dict(
            color=TEXT_2,
        ),

        ticks="",
    )

    return fig


# ============================================================
# GLOWING LINES
# ============================================================

def add_glow_line(
    fig,
    x,
    y,
    name,
    color,
    *,
    dash="solid",
    fill=False,
    small=False,
):
    # Outer glow
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,

            mode="lines",

            showlegend=False,

            hoverinfo="skip",

            opacity=.07,

            line=dict(
                color=color,

                width=(
                    10
                    if not small
                    else 7
                ),

                shape="spline",

                smoothing=1.05,

                dash=dash,
            ),
        )
    )

    # Middle glow
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,

            mode="lines",

            showlegend=False,

            hoverinfo="skip",

            opacity=.15,

            line=dict(
                color=color,

                width=(
                    6
                    if not small
                    else 4
                ),

                shape="spline",

                smoothing=1.05,

                dash=dash,
            ),
        )
    )

    # Actual line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,

            mode=
                "lines+markers",

            name=name,

            line=dict(
                color=color,

                width=(
                    2.7
                    if not small
                    else 2.1
                ),

                shape="spline",

                smoothing=1.05,

                dash=dash,
            ),

            marker=dict(
                size=(
                    6
                    if not small
                    else 4
                ),

                color="white",

                line=dict(
                    color=color,

                    width=2,
                ),
            ),

            fill=(
                "tozeroy"
                if fill
                else None
            ),

            fillcolor=(
                hex_to_rgba(
                    color,
                    .08,
                )
                if fill
                else None
            ),

            hovertemplate=(
                "<b>%{x}</b><br>"
                + name
                + ": %{y:.1%}"
                "<extra></extra>"
            ),
        )
    )


# ============================================================
# CHART ANIMATION
# ============================================================

def add_entrance_animation(
    fig,
    steps=20,
):
    animations = []

    for index, trace in enumerate(
        fig.data
    ):
        # Bar chart
        if trace.type == "bar":

            if trace.orientation == "h":

                final_values = [
                    float(value)
                    for value
                    in trace.x
                ]

                animations.append(
                    (
                        index,
                        "x",
                        final_values,
                        "bar",
                    )
                )

                trace.x = [
                    0
                    for _
                    in final_values
                ]

            else:

                final_values = [
                    float(value)
                    for value
                    in trace.y
                ]

                animations.append(
                    (
                        index,
                        "y",
                        final_values,
                        "bar",
                    )
                )

                trace.y = [
                    0
                    for _
                    in final_values
                ]

        # Line chart
        elif (
            trace.type == "scatter"
            and trace.y is not None
            and "lines"
            in (
                trace.mode
                or ""
            )
        ):
            # Random baseline stays fixed
            if trace.name == "Random":
                continue

            final_values = [
                float(value)
                for value
                in trace.y
            ]

            animations.append(
                (
                    index,
                    "y",
                    final_values,
                    "scatter",
                )
            )

            trace.y = [
                0
                for _
                in final_values
            ]

    if not animations:
        return fig

    frames = []

    for step in range(
        1,
        steps + 1,
    ):
        progress = (
            step / steps
        )

        # Smooth ease-out
        eased = (
            1
            - (
                1 - progress
            ) ** 3
        )

        frame_data = []
        trace_indexes = []

        for (
            index,
            axis,
            final_values,
            trace_type,
        ) in animations:

            values = [
                value * eased
                for value
                in final_values
            ]

            if trace_type == "bar":

                if axis == "x":
                    update = (
                        go.Bar(
                            x=values
                        )
                    )

                else:
                    update = (
                        go.Bar(
                            y=values
                        )
                    )

            else:
                update = (
                    go.Scatter(
                        y=values
                    )
                )

            frame_data.append(
                update
            )

            trace_indexes.append(
                index
            )

        frames.append(
            go.Frame(
                data=frame_data,

                traces=
                    trace_indexes,

                name=
                    f"frame_{step}",
            )
        )

    fig.frames = frames

    return fig


def render_animated_chart(
    fig,
    height,
    frame_duration=36,
):
    post_script = f"""
        const graph =
            document.getElementById('{{plot_id}}');

        setTimeout(function() {{

            Plotly.animate(
                graph,
                null,
                {{
                    frame: {{
                        duration:
                            {frame_duration},

                        redraw:
                            false
                    }},

                    transition: {{
                        duration:
                            {frame_duration},

                        easing:
                            'cubic-out'
                    }},

                    mode:
                        'immediate'
                }}
            );

        }}, 180);
    """

    html = fig.to_html(
        full_html=False,

        include_plotlyjs=
            "cdn",

        auto_play=False,

        config=
            PLOT_CONFIG,

        post_script=
            post_script,
    )

    components.html(
        html,

        height=
            height + 16,

        scrolling=False,
    )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def get_data():
    raw = (
        load_raw_data()
    )

    cleaned = (
        clean_data(
            raw
        )
    )

    return (
        raw,
        cleaned,
    )


raw_df, df = (
    get_data()
)


(
    X,
    y,
    numeric_features,
    categorical_features,
) = split_features_target(
    df
)


preprocessor = (
    build_preprocessor(
        numeric_features,
        categorical_features,
    )
)


# ============================================================
# CACHED MODELS
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def cached_model(
    model_name,
):
    model = (
        available_models()[
            model_name
        ]
    )

    return (
        train_and_evaluate(
            X,
            y,

            preprocessor,

            model,
        )
    )


@st.cache_data(
    show_spinner=False
)
def cached_comparison():
    return (
        compare_models(
            X,
            y,

            preprocessor,

            available_models(),
        )
    )


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

@st.cache_data(
    show_spinner=False
)
def permutation_explanation(
    model_name,
):
    """
    Used for models such as KNN and Naive Bayes
    which do not expose coef_ or feature_importances_.
    """

    result = (
        cached_model(
            model_name
        )
    )

    pipeline = (
        result[
            "pipeline"
        ]
    )

    # --------------------------------------------------------
    # Sample data to keep KNN permutation importance fast.
    # --------------------------------------------------------

    sample_size = min(
        2500,
        len(X),
    )

    X_sample = (
        X.sample(
            n=sample_size,
            random_state=42,
        )
    )

    # Keep target aligned with X_sample
    if hasattr(
        y,
        "loc",
    ):
        y_sample = (
            y.loc[
                X_sample.index
            ]
        )

    else:
        y_series = (
            pd.Series(
                y,
                index=X.index,
            )
        )

        y_sample = (
            y_series.loc[
                X_sample.index
            ]
        )

    importance = (
        permutation_importance(
            pipeline,

            X_sample,
            y_sample,

            scoring=
                "roc_auc",

            n_repeats=3,

            random_state=42,

            n_jobs=-1,
        )
    )

    explanation = (
        pd.DataFrame(
            {
                "Feature":
                    X.columns,

                "Importance":
                    importance.importances_mean,

                "ImportanceStd":
                    importance.importances_std,
            }
        )
    )

    explanation[
        "AbsoluteImportance"
    ] = (
        explanation[
            "Importance"
        ].abs()
    )

    explanation = (
        explanation

        .sort_values(
            "AbsoluteImportance",
            ascending=False,
        )

        .reset_index(
            drop=True
        )
    )

    return explanation


# ============================================================
# MONTHLY DATA
# ============================================================

MONTH_ORDER = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]


monthly = (
    df.groupby(
        "month"
    )

    .agg(
        customers=(
            "y",
            "size",
        ),

        subscription=(
            "y",
            "mean",
        ),

        balance=(
            "balance",
            "median",
        ),

        previous=(
            "previous",
            "mean",
        ),
    )

    .reindex(
        MONTH_ORDER
    )

    .dropna()

    .reset_index()
)


monthly[
    "label"
] = (
    monthly[
        "month"
    ].str.title()
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    (
        "<div class='sidebar-brand'>"

        "<div class='brand-logo'>"

        "<div class='logo-bars'>"

        "<span></span>"
        "<span></span>"
        "<span></span>"
        "<span></span>"

        "</div>"

        "</div>"

        "<div>"

        "<div class='brand-name'>"
        "BankLens"
        "</div>"

        "<div class='brand-caption'>"
        "Marketing Analytics"
        "</div>"

        "</div>"

        "</div>"
    ),
    unsafe_allow_html=True,
)


st.sidebar.markdown(
    (
        "<div class='sidebar-label'>"
        "Workspace"
        "</div>"
    ),
    unsafe_allow_html=True,
)


st.sidebar.markdown(
    (
        "<div class='sidebar-card'>"

        "<div class='sidebar-card-title'>"
        "Campaign Dataset"
        "</div>"

        "<div class='sidebar-card-text'>"

        f"{len(df):,} customer records"

        "<br>"

        f"{X.shape[1]} modelling features"

        "<br>"

        f"{len(available_models())} models"

        "</div>"

        "</div>"
    ),
    unsafe_allow_html=True,
)


st.sidebar.markdown(
    (
        "<div class='sidebar-card'>"

        "<div style='"
        "display:flex;"
        "align-items:center;"
        "gap:8px;"
        "margin-bottom:7px;"
        "color:#4A8EF6;"
        "'>"

        f"{icon_svg('trend',18,BLUE)}"

        "<b style='font-size:11px;'>"
        "Analytics Workspace"
        "</b>"

        "</div>"

        "<div class='sidebar-card-text'>"

        "Explore customer behaviour, "
        "compare models and estimate "
        "subscription probability."

        "</div>"

        "</div>"
    ),
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

header_left, header_right = (
    st.columns(
        [
            5.5,
            1.5,
        ]
    )
)


with header_left:

    st.markdown(
        (
            "<div class='page-title'>"
            "Bank Marketing Analytics"
            "</div>"

            "<div class='page-subtitle'>"

            "Customer behaviour, "
            "campaign performance "
            "and predictive modelling"

            "</div>"
        ),
        unsafe_allow_html=True,
    )


with header_right:

    st.markdown(
        (
            "<div style='"
            "display:flex;"
            "justify-content:flex-end;"
            "align-items:center;"
            "padding-top:3px;"
            "'>"

            "<div class='header-chip'>"

            f"{icon_svg('calendar',15,'#62728D')}"

            " Last 12 months"

            "</div>"

            "</div>"
        ),
        unsafe_allow_html=True,
    )


st.write("")


# ============================================================
# SEARCH
# ============================================================

search = (
    st.text_input(
        "Search",

        placeholder=(
            "Search features, models, "
            "metrics or insights..."
        ),

        label_visibility=
            "collapsed",
    )
)


if search:

    search_items = (
        X.columns.tolist()

        + list(
            available_models().keys()
        )

        + [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
        ]
    )

    matches = [
        item

        for item
        in search_items

        if (
            search.lower()
            in item.lower()
        )
    ]

    if matches:
        st.caption(
            "Matches: "
            + " · ".join(
                matches[:8]
            )
        )


st.write("")


# ============================================================
# KPI ROW
# ============================================================

k1, k2, k3, k4 = (
    st.columns(
        4
    )
)


with k1:

    kpi_card(
        "Customers Analysed",

        f"{len(df):,}",

        "Campaign dataset",

        CYAN,

        "#F0FDFD",

        "#E3F9FA",

        icon_svg(
            "users",
            19,
            CYAN,
        ),

        monthly[
            "customers"
        ].tolist(),
    )


with k2:

    kpi_card(
        "Subscription Rate",

        f"{y.mean():.1%}",

        "Positive response",

        BLUE,

        "#EFF5FF",

        "#E7F0FF",

        icon_svg(
            "percent",
            19,
            BLUE,
        ),

        monthly[
            "subscription"
        ].tolist(),
    )


with k3:

    kpi_card(
        "Median Balance",

        f"{df['balance'].median():,.0f}",

        "Customer balance",

        PURPLE,

        "#F7F3FF",

        "#F0E9FF",

        icon_svg(
            "database",
            18,
            PURPLE,
        ),

        monthly[
            "balance"
        ].tolist(),
    )


with k4:

    kpi_card(
        "Previous Contacts",

        f"{df['previous'].mean():.2f}",

        "Average per customer",

        AMBER,

        "#FFF8E9",

        "#FFF1D0",

        icon_svg(
            "phone",
            18,
            AMBER,
        ),

        monthly[
            "previous"
        ].tolist(),
    )


st.write("")


# ============================================================
# NAVIGATION
# ============================================================

(
    overview_tab,
    data_tab,
    model_tab,
    comparison_tab,
    prediction_tab,
) = st.tabs(
    [
        "📈  Overview",
        "🗂️  Data Explorer",
        "🧪  Model Lab",
        "📊  Compare Models",
        "👥  Customer Prediction",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with overview_tab:

    st.write("")

    (
        main_left,
        main_right,
    ) = st.columns(
        [
            2.2,
            1,
        ]
    )


    # ========================================================
    # RESPONSE PROFILE
    # ========================================================

    with main_left:

        with st.container(
            border=True
        ):

            section_header(
                "Campaign Analytics",

                "Customer response profile by month",

                (
                    "Compare response, cellular contact "
                    "and previous campaign success."
                ),
            )

            profile = (
                df.assign(
                    cellular_rate=(
                        (
                            df[
                                "contact"
                            ]
                            == "cellular"
                        )
                        .astype(
                            int
                        )
                    ),

                    previous_success=(
                        (
                            df[
                                "poutcome"
                            ]
                            == "success"
                        )
                        .astype(
                            int
                        )
                    ),
                )

                .groupby(
                    "month"
                )

                .agg(
                    subscription=(
                        "y",
                        "mean",
                    ),

                    cellular=(
                        "cellular_rate",
                        "mean",
                    ),

                    previous_success=(
                        "previous_success",
                        "mean",
                    ),
                )

                .reindex(
                    MONTH_ORDER
                )

                .dropna()

                .reset_index()
            )

            profile[
                "label"
            ] = (
                profile[
                    "month"
                ].str.title()
            )

            fig = (
                go.Figure()
            )

            add_glow_line(
                fig,

                profile[
                    "label"
                ],

                profile[
                    "subscription"
                ],

                "Subscription rate",

                BLUE,

                fill=True,
            )

            add_glow_line(
                fig,

                profile[
                    "label"
                ],

                profile[
                    "cellular"
                ],

                "Cellular contact",

                PURPLE,
            )

            add_glow_line(
                fig,

                profile[
                    "label"
                ],

                profile[
                    "previous_success"
                ],

                "Previous success",

                CYAN,

                dash="dash",
            )

            fig = (
                style_chart(
                    fig,

                    height=390,
                )
            )

            fig.update_layout(
                hovermode=
                    "x unified"
            )

            fig.update_yaxes(
                title="Rate",

                tickformat=".0%",

                range=[
                    0,
                    1.02,
                ],
            )

            fig = (
                add_entrance_animation(
                    fig,

                    steps=24,
                )
            )

            render_animated_chart(
                fig,

                height=390,

                frame_duration=32,
            )


    # ========================================================
    # MODELLING READINESS
    # ========================================================

    with main_right:

        with st.container(
            border=True
        ):

            section_header(
                "Dataset Health",

                "Modelling readiness",

                (
                    "Dataset quality and "
                    "preprocessing status."
                ),
            )

            ring = (
                go.Figure(
                    go.Pie(
                        values=[
                            25,
                            25,
                            25,
                            25,
                        ],

                        hole=.70,

                        sort=False,

                        marker=dict(
                            colors=[
                                CYAN,
                                BLUE,
                                PURPLE,
                                PINK,
                            ],

                            line=dict(
                                color="white",

                                width=4,
                            ),
                        ),

                        textinfo=
                            "none",

                        hoverinfo=
                            "skip",
                    )
                )
            )

            ring.update_layout(
                height=225,

                paper_bgcolor=
                    "rgba(0,0,0,0)",

                plot_bgcolor=
                    "rgba(0,0,0,0)",

                showlegend=False,

                margin=dict(
                    l=10,
                    r=10,
                    t=5,
                    b=5,
                ),

                annotations=[
                    dict(
                        x=.5,
                        y=.5,

                        text=(
                            f"<b>{y.mean():.1%}</b>"

                            "<br>"

                            "<span "
                            "style='font-size:10px;"
                            "color:#7E899C'>"

                            "Subscribed"

                            "</span>"
                        ),

                        showarrow=False,

                        font=dict(
                            color=TEXT,

                            size=23,
                        ),
                    )
                ],
            )

            st.plotly_chart(
                ring,

                use_container_width=True,

                config={
                    "displayModeBar":
                        False,

                    "responsive":
                        True,
                },

                key=
                    "readiness",
            )

            missing = int(
                df.isna()
                .sum()
                .sum()
            )

            status_row(
                "Target variable",
                "Ready",
                GREEN,
            )

            status_row(
                "Data leakage",
                "Removed",
                GREEN,
            )

            status_row(
                "Missing values",

                str(
                    missing
                ),

                (
                    GREEN
                    if missing == 0
                    else AMBER
                ),
            )

            status_row(
                "Class balance",
                "Imbalanced",
                AMBER,
            )


    # ========================================================
    # LOWER ANALYTICS
    # ========================================================

    st.write("")

    comparison = (
        cached_comparison()
    )

    preview_name = (
        "Logistic Regression"
    )

    if (
        preview_name
        not in available_models()
    ):
        preview_name = (
            list(
                available_models()
                .keys()
            )[0]
        )

    preview = (
        cached_model(
            preview_name
        )
    )

    (
        c1,
        c2,
        c3,
    ) = st.columns(
        [
            1.2,
            1,
            1,
        ]
    )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    with c1:

        with st.container(
            border=True
        ):

            section_header(
                "Performance",

                "Model comparison",
            )

            fig = (
                go.Figure()
            )

            metric_colors = {
                "Precision":
                    BLUE,

                "Recall":
                    CYAN,

                "F1":
                    PURPLE,

                "ROC-AUC":
                    PINK,
            }

            for (
                metric,
                color,
            ) in metric_colors.items():

                fig.add_trace(
                    go.Bar(
                        x=comparison[
                            "Model"
                        ],

                        y=comparison[
                            metric
                        ],

                        name=metric,

                        marker=dict(
                            color=color,

                            line=dict(
                                color=
                                    hex_to_rgba(
                                        color,
                                        .85,
                                    ),

                                width=1,
                            ),
                        ),

                        opacity=.86,

                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            + metric
                            + ": %{y:.3f}"
                            "<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(
                barmode=
                    "group",

                bargap=.24,

                bargroupgap=.08,
            )

            fig = (
                style_chart(
                    fig,

                    height=250,

                    small=True,
                )
            )

            fig.update_yaxes(
                range=[
                    0,
                    1,
                ]
            )

            fig = (
                add_entrance_animation(
                    fig,

                    steps=20,
                )
            )

            render_animated_chart(
                fig,

                height=250,

                frame_duration=36,
            )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    with c2:

        with st.container(
            border=True
        ):

            section_header(
                "Classification",

                "Confusion matrix",
            )

            cm = (
                confusion_matrix_dataframe(
                    preview[
                        "confusion_matrix"
                    ]
                )
            )

            fig = (
                go.Figure(
                    go.Heatmap(
                        z=cm.values,

                        x=[
                            "No",
                            "Yes",
                        ],

                        y=[
                            "No",
                            "Yes",
                        ],

                        text=
                            cm.values,

                        texttemplate=
                            "<b>%{text:,}</b>",

                        textfont=dict(
                            size=11,
                        ),

                        colorscale=[
                            [
                                0,
                                "#F4F8FD",
                            ],
                            [
                                .38,
                                "#B7DAF9",
                            ],
                            [
                                .70,
                                "#6BADEB",
                            ],
                            [
                                1,
                                CYAN,
                            ],
                        ],

                        showscale=False,
                    )
                )
            )

            fig = (
                style_chart(
                    fig,

                    height=250,

                    legend=False,

                    small=True,
                )
            )

            fig.update_yaxes(
                autorange=
                    "reversed"
            )

            st.plotly_chart(
                fig,

                use_container_width=True,

                config=
                    PLOT_CONFIG,

                key=
                    "overview_cm",
            )


    # ========================================================
    # ROC
    # ========================================================

    with c3:

        with st.container(
            border=True
        ):

            section_header(
                "Model Quality",

                "ROC curve",
            )

            roc = (
                preview[
                    "roc_data"
                ]
            )

            fig = (
                go.Figure()
            )

            add_glow_line(
                fig,

                roc[
                    "False Positive Rate"
                ],

                roc[
                    "True Positive Rate"
                ],

                (
                    "Model "
                    f"(AUC="
                    f"{preview['metrics']['ROC-AUC']:.2f})"
                ),

                BLUE,

                fill=True,

                small=True,
            )

            fig.add_trace(
                go.Scatter(
                    x=[
                        0,
                        1,
                    ],

                    y=[
                        0,
                        1,
                    ],

                    mode=
                        "lines",

                    name=
                        "Random",

                    line=dict(
                        color=
                            PURPLE,

                        dash=
                            "dash",

                        width=2,
                    ),
                )
            )

            fig = (
                style_chart(
                    fig,

                    height=250,

                    small=True,
                )
            )

            fig.update_xaxes(
                range=[
                    0,
                    1,
                ]
            )

            fig.update_yaxes(
                range=[
                    0,
                    1,
                ]
            )

            fig = (
                add_entrance_animation(
                    fig,

                    steps=22,
                )
            )

            render_animated_chart(
                fig,

                height=250,

                frame_duration=32,
            )


# ============================================================
# DATA EXPLORER
# ============================================================

with data_tab:

    st.write("")

    section_header(
        "Data Analytics",

        "Customer feature explorer",

        (
            "Inspect individual variables "
            "and their relationship "
            "with subscription."
        ),
    )

    default_index = (
        X.columns.tolist()
        .index(
            "age"
        )

        if (
            "age"
            in X.columns
        )

        else 0
    )

    feature = (
        st.selectbox(
            "Feature",

            X.columns.tolist(),

            index=
                default_index,

            key=
                "feature_selector",
        )
    )

    with st.expander(
        "View dataset sample"
    ):

        st.dataframe(
            df.head(
                30
            ),

            use_container_width=True,
        )


    # ========================================================
    # NUMERICAL
    # ========================================================

    if (
        feature
        in numeric_features
    ):

        with st.container(
            border=True
        ):

            section_header(
                "Distribution Analysis",

                feature
                .replace(
                    "_",
                    " "
                )
                .title(),

                (
                    "Subscribers vs "
                    "non-subscribers."
                ),
            )

            fig = (
                go.Figure()
            )

            for (
                target,
                name,
                color,
            ) in [
                (
                    0,
                    "Not subscribed",
                    BLUE,
                ),
                (
                    1,
                    "Subscribed",
                    PINK,
                ),
            ]:

                subset = (
                    df.loc[
                        df[
                            "y"
                        ] == target,

                        feature,
                    ]
                )

                fig.add_trace(
                    go.Histogram(
                        x=subset,

                        name=name,

                        opacity=.60,

                        marker=dict(
                            color=color,

                            line=dict(
                                color=
                                    hex_to_rgba(
                                        color,
                                        .88,
                                    ),

                                width=1,
                            ),
                        ),

                        hovertemplate=(
                            "<b>"
                            + name
                            + "</b><br>"
                            + feature
                            + ": %{x}<br>"
                            "Customers: %{y:,}"
                            "<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(
                barmode=
                    "overlay"
            )

            fig = (
                style_chart(
                    fig,

                    height=420,
                )
            )

            fig.update_xaxes(
                title=(
                    feature
                    .replace(
                        "_",
                        " "
                    )
                    .title()
                )
            )

            fig.update_yaxes(
                title=
                    "Customers"
            )

            st.plotly_chart(
                fig,

                use_container_width=True,

                config=
                    PLOT_CONFIG,

                key=
                    f"numeric_{feature}",
            )


    # ========================================================
    # CATEGORICAL
    # ========================================================

    else:

        with st.container(
            border=True
        ):

            section_header(
                "Conversion Analysis",

                (
                    "Subscription rate by "
                    + feature
                    .replace(
                        "_",
                        " "
                    )
                    .title()
                ),
            )

            summary = (
                df.groupby(
                    feature
                )[
                    "y"
                ]

                .agg(
                    subscription_rate=
                        "mean",

                    customers=
                        "count",
                )

                .reset_index()

                .sort_values(
                    "subscription_rate",

                    ascending=True,
                )
            )

            palette = [
                BLUE,
                CYAN,
                PURPLE,
                PINK,

                "#73A6F3",
                "#6ACFD4",
                "#A780E9",
                "#EE89B6",
            ]

            colors = [
                palette[
                    index
                    % len(
                        palette
                    )
                ]

                for index
                in range(
                    len(
                        summary
                    )
                )
            ]

            chart_height = max(
                350,

                len(
                    summary
                )
                * 36,
            )

            fig = (
                go.Figure(
                    go.Bar(
                        x=summary[
                            "subscription_rate"
                        ],

                        y=summary[
                            feature
                        ],

                        orientation=
                            "h",

                        customdata=
                            summary[
                                "customers"
                            ],

                        marker=dict(
                            color=
                                colors,

                            line=dict(
                                color=
                                    "white",

                                width=1,
                            ),
                        ),

                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Subscription rate: "
                            "%{x:.1%}<br>"
                            "Customers: "
                            "%{customdata:,}"
                            "<extra></extra>"
                        ),
                    )
                )
            )

            fig.update_layout(
                bargap=.28,

                barcornerradius=8,
            )

            fig = (
                style_chart(
                    fig,

                    height=
                        chart_height,

                    legend=False,
                )
            )

            fig.update_xaxes(
                title=
                    "Subscription rate",

                tickformat=
                    ".0%",
            )

            fig.update_yaxes(
                automargin=True,

                title="",
            )

            fig = (
                add_entrance_animation(
                    fig,

                    steps=22,
                )
            )

            render_animated_chart(
                fig,

                height=
                    chart_height,

                frame_duration=35,
            )


# ============================================================
# MODEL LAB
# ============================================================

with model_tab:

    st.write("")

    section_header(
        "Machine Learning",

        "Model laboratory",

        (
            "Train one model "
            "and inspect its "
            "predictive behaviour."
        ),
    )

    model_name = (
        st.selectbox(
            "Model",

            list(
                available_models()
                .keys()
            ),

            key=
                "model_lab_selector",
        )
    )

    train_button = (
        st.button(
            "Train selected model",

            key=
                "train_model_button",
        )
    )

    if train_button:

        st.session_state[
            "trained_model"
        ] = (
            model_name
        )

    if (
        "trained_model"
        in st.session_state
    ):

        trained_name = (
            st.session_state[
                "trained_model"
            ]
        )

        result = (
            cached_model(
                trained_name
            )
        )

        metrics = (
            result[
                "metrics"
            ]
        )


        # ====================================================
        # METRICS
        # ====================================================

        cols = (
            st.columns(
                5
            )
        )

        metric_values = [
            (
                "Accuracy",
                metrics[
                    "Accuracy"
                ],
            ),
            (
                "Precision",
                metrics[
                    "Precision"
                ],
            ),
            (
                "Recall",
                metrics[
                    "Recall"
                ],
            ),
            (
                "F1",
                metrics[
                    "F1"
                ],
            ),
            (
                "ROC-AUC",
                metrics[
                    "ROC-AUC"
                ],
            ),
        ]

        for (
            column,
            (
                label,
                value,
            ),
        ) in zip(
            cols,
            metric_values,
        ):

            column.metric(
                label,

                f"{value:.3f}",
            )


        st.write("")

        (
            left,
            right,
        ) = (
            st.columns(
                2
            )
        )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        with left:

            with st.container(
                border=True
            ):

                section_header(
                    "Classification",

                    "Confusion matrix",

                    (
                        "Actual classes compared "
                        "with model predictions."
                    ),
                )

                cm = (
                    confusion_matrix_dataframe(
                        result[
                            "confusion_matrix"
                        ]
                    )
                )

                fig = (
                    go.Figure(
                        go.Heatmap(
                            z=
                                cm.values,

                            x=[
                                "Predicted No",
                                "Predicted Yes",
                            ],

                            y=[
                                "Actual No",
                                "Actual Yes",
                            ],

                            text=
                                cm.values,

                            texttemplate=
                                "<b>%{text:,}</b>",

                            textfont=dict(
                                size=12,

                                color=
                                    "#263244",
                            ),

                            colorscale=[
                                [
                                    0,
                                    "#F5F9FE",
                                ],
                                [
                                    .30,
                                    "#C9E1F8",
                                ],
                                [
                                    .65,
                                    "#72AFE8",
                                ],
                                [
                                    1,
                                    CYAN,
                                ],
                            ],

                            showscale=False,

                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "%{x}<br>"
                                "Customers: %{z:,}"
                                "<extra></extra>"
                            ),
                        )
                    )
                )

                fig = (
                    style_chart(
                        fig,

                        height=380,

                        legend=False,
                    )
                )

                fig.update_yaxes(
                    autorange=
                        "reversed",

                    automargin=True,
                )

                fig.update_xaxes(
                    automargin=True,
                )

                st.plotly_chart(
                    fig,

                    use_container_width=True,

                    config=
                        PLOT_CONFIG,

                    key=
                        f"model_lab_cm_{trained_name}",
                )


        # ====================================================
        # ROC
        # ====================================================

        with right:

            with st.container(
                border=True
            ):

                section_header(
                    "Model Quality",

                    "ROC curve",

                    (
                        "Sensitivity versus false-positive "
                        "rate across decision thresholds."
                    ),
                )

                roc = (
                    result[
                        "roc_data"
                    ]
                )

                fig = (
                    go.Figure()
                )

                add_glow_line(
                    fig,

                    roc[
                        "False Positive Rate"
                    ],

                    roc[
                        "True Positive Rate"
                    ],

                    (
                        f"{trained_name} "
                        f"(AUC="
                        f"{metrics['ROC-AUC']:.2f})"
                    ),

                    BLUE,

                    fill=True,
                )

                fig.add_trace(
                    go.Scatter(
                        x=[
                            0,
                            1,
                        ],

                        y=[
                            0,
                            1,
                        ],

                        mode=
                            "lines",

                        name=
                            "Random",

                        line=dict(
                            color=
                                PURPLE,

                            dash=
                                "dash",

                            width=2,
                        ),

                        hoverinfo=
                            "skip",
                    )
                )

                fig = (
                    style_chart(
                        fig,

                        height=380,
                    )
                )

                fig.update_xaxes(
                    title=
                        "False Positive Rate",

                    range=[
                        0,
                        1,
                    ],
                )

                fig.update_yaxes(
                    title=
                        "True Positive Rate",

                    range=[
                        0,
                        1,
                    ],
                )

                fig = (
                    add_entrance_animation(
                        fig,

                        steps=24,
                    )
                )

                render_animated_chart(
                    fig,

                    height=380,

                    frame_duration=32,
                )


        # ====================================================
        # MODEL INTERPRETABILITY
        # ====================================================

        try:

            (
                interpretation_type,
                explanation,
            ) = explain_model(
                result[
                    "pipeline"
                ]
            )

        except (
            AttributeError,
            ValueError,
            TypeError,
        ):

            interpretation_type = None
            explanation = None


        # ----------------------------------------------------
        # FALLBACK FOR KNN / NAIVE BAYES
        # ----------------------------------------------------

        if explanation is None:

            interpretation_type = (
                "permutation"
            )

            with st.spinner(
                (
                    "Calculating permutation "
                    "feature importance..."
                )
            ):

                explanation = (
                    permutation_explanation(
                        trained_name
                    )
                )


        # ====================================================
        # INTERPRETABILITY CHART
        # ====================================================

        if explanation is not None:

            with st.container(
                border=True
            ):

                if (
                    interpretation_type
                    == "permutation"
                ):

                    interpretation_description = (
                        "Feature influence estimated by "
                        "measuring how much ROC-AUC changes "
                        "when each feature is shuffled."
                    )

                elif (
                    interpretation_type
                    == "coefficient"
                ):

                    interpretation_description = (
                        "Model coefficients showing the "
                        "strongest positive and negative "
                        "predictive relationships."
                    )

                else:

                    interpretation_description = (
                        "Features with the strongest influence "
                        "on this model's predictions."
                    )

                section_header(
                    "Interpretability",

                    "Most influential features",

                    interpretation_description,
                )


                # ------------------------------------------------
                # VALUE COLUMN
                # ------------------------------------------------

                if (
                    interpretation_type
                    == "coefficient"
                ):
                    value_column = (
                        "Coefficient"
                    )

                else:
                    value_column = (
                        "Importance"
                    )


                # ------------------------------------------------
                # CLEAN DATA
                # ------------------------------------------------

                chart_data = (
                    explanation
                    .copy()
                )

                chart_data[
                    "Feature"
                ] = (
                    chart_data[
                        "Feature"
                    ]
                    .apply(
                        clean_feature_name
                    )
                )


                # ------------------------------------------------
                # RANK STRONGEST FEATURES
                # ------------------------------------------------

                if (
                    interpretation_type
                    == "coefficient"
                ):

                    chart_data[
                        "_strength"
                    ] = (
                        chart_data[
                            value_column
                        ].abs()
                    )

                elif (
                    "AbsoluteImportance"
                    in chart_data.columns
                ):

                    chart_data[
                        "_strength"
                    ] = (
                        chart_data[
                            "AbsoluteImportance"
                        ]
                    )

                else:

                    chart_data[
                        "_strength"
                    ] = (
                        chart_data[
                            value_column
                        ].abs()
                    )


                chart_data = (
                    chart_data

                    .sort_values(
                        "_strength",

                        ascending=False,
                    )

                    .head(
                        15
                    )

                    .sort_values(
                        value_column,

                        ascending=True,
                    )
                )


                # ------------------------------------------------
                # COLORS
                # ------------------------------------------------

                if (
                    interpretation_type
                    == "coefficient"
                ):

                    colors = [
                        (
                            GREEN
                            if value >= 0
                            else PINK
                        )

                        for value
                        in chart_data[
                            value_column
                        ]
                    ]

                elif (
                    interpretation_type
                    == "permutation"
                ):

                    positive_palette = [
                        BLUE,
                        PURPLE,
                        CYAN,
                    ]

                    colors = []

                    for (
                        index,
                        value,
                    ) in enumerate(
                        chart_data[
                            value_column
                        ]
                    ):

                        if value < 0:

                            colors.append(
                                PINK
                            )

                        else:

                            colors.append(
                                positive_palette[
                                    index
                                    % len(
                                        positive_palette
                                    )
                                ]
                            )

                else:

                    colors = [
                        (
                            BLUE
                            if index % 2 == 0
                            else PURPLE
                        )

                        for index
                        in range(
                            len(
                                chart_data
                            )
                        )
                    ]


                # ------------------------------------------------
                # HEIGHT
                # ------------------------------------------------

                chart_height = max(
                    450,

                    len(
                        chart_data
                    )
                    * 32,
                )


                # ------------------------------------------------
                # PLOT
                # ------------------------------------------------

                fig = (
                    go.Figure(
                        go.Bar(
                            x=chart_data[
                                value_column
                            ],

                            y=chart_data[
                                "Feature"
                            ],

                            orientation=
                                "h",

                            marker=dict(
                                color=
                                    colors,

                                line=dict(
                                    color=
                                        "white",

                                    width=1,
                                ),
                            ),

                            customdata=(
                                chart_data[
                                    "_strength"
                                ]
                            ),

                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "Value: %{x:.4f}"
                                "<extra></extra>"
                            ),
                        )
                    )
                )

                fig.update_layout(
                    bargap=.27,

                    barcornerradius=7,
                )

                fig = (
                    style_chart(
                        fig,

                        height=
                            chart_height,

                        legend=False,
                    )
                )


                # =================================================
                # FIX CLIPPED Y-AXIS LABELS
                # =================================================

                longest_label = max(
                    len(
                        str(
                            feature_name
                        )
                    )

                    for feature_name
                    in chart_data[
                        "Feature"
                    ]
                )

                left_margin = min(
                    max(
                        160,

                        int(
                            longest_label
                            * 8.5
                        ),
                    ),

                    320,
                )

                fig.update_layout(
                    margin=dict(
                        l=
                            left_margin,

                        r=35,

                        t=30,

                        b=55,
                    )
                )

                fig.update_yaxes(
                    automargin=True,

                    ticklabelposition=
                        "outside",

                    tickfont=dict(
                        size=11,

                        color=
                            TEXT_2,
                    ),

                    title="",
                )


                # ------------------------------------------------
                # X AXIS
                # ------------------------------------------------

                if (
                    interpretation_type
                    == "coefficient"
                ):

                    fig.update_xaxes(
                        title=
                            "Model coefficient"
                    )

                    fig.add_vline(
                        x=0,

                        line_width=1,

                        line_dash=
                            "dash",

                        line_color=
                            "#A9B5C7",
                    )

                elif (
                    interpretation_type
                    == "permutation"
                ):

                    fig.update_xaxes(
                        title=(
                            "Change in ROC-AUC "
                            "when feature is shuffled"
                        )
                    )

                    fig.add_vline(
                        x=0,

                        line_width=1,

                        line_dash=
                            "dash",

                        line_color=
                            "#A9B5C7",
                    )

                else:

                    fig.update_xaxes(
                        title=
                            "Feature importance"
                    )


                # ------------------------------------------------
                # ANIMATION
                # ------------------------------------------------

                fig = (
                    add_entrance_animation(
                        fig,

                        steps=22,
                    )
                )

                render_animated_chart(
                    fig,

                    height=
                        chart_height,

                    frame_duration=35,
                )


                # ------------------------------------------------
                # MODEL-SPECIFIC NOTE
                # ------------------------------------------------

                if (
                    interpretation_type
                    == "permutation"
                ):

                    st.caption(
                        (
                            f"{trained_name} does not expose "
                            "direct coefficients or tree-based "
                            "feature importance. Therefore, this "
                            "chart uses permutation importance: "
                            "each feature is shuffled and the "
                            "change in ROC-AUC is measured."
                        )
                    )


# ============================================================
# MODEL COMPARISON
# ============================================================

with comparison_tab:

    st.write("")

    section_header(
        "Benchmark",

        "Model performance comparison",

        (
            "Compare classification "
            "performance across statistical "
            "and machine-learning models."
        ),
    )

    if st.button(
        "Run model comparison",

        key=
            "compare_models_button",
    ):

        st.session_state[
            "show_comparison"
        ] = True


    if st.session_state.get(
        "show_comparison",
        False,
    ):

        comparison = (
            cached_comparison()
        )

        st.dataframe(
            comparison
            .style
            .format(
                {
                    "Accuracy":
                        "{:.3f}",

                    "Precision":
                        "{:.3f}",

                    "Recall":
                        "{:.3f}",

                    "F1":
                        "{:.3f}",

                    "ROC-AUC":
                        "{:.3f}",
                }
            ),

            use_container_width=True,
        )

        st.write("")

        with st.container(
            border=True
        ):

            section_header(
                "Performance",

                "Classification metric comparison",

                (
                    "Precision, recall, F1 and ROC-AUC "
                    "across the available models."
                ),
            )

            fig = (
                go.Figure()
            )

            colors = {
                "Precision":
                    BLUE,

                "Recall":
                    CYAN,

                "F1":
                    PURPLE,

                "ROC-AUC":
                    PINK,
            }

            for (
                metric,
                color,
            ) in colors.items():

                fig.add_trace(
                    go.Bar(
                        x=comparison[
                            "Model"
                        ],

                        y=comparison[
                            metric
                        ],

                        name=metric,

                        marker=dict(
                            color=
                                color,

                            line=dict(
                                color=
                                    "white",

                                width=.8,
                            ),
                        ),

                        opacity=.87,

                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            + metric
                            + ": %{y:.3f}"
                            "<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(
                barmode=
                    "group",

                bargap=.23,

                bargroupgap=.08,

                barcornerradius=5,
            )

            fig = (
                style_chart(
                    fig,

                    height=420,
                )
            )

            fig.update_yaxes(
                title=
                    "Score",

                range=[
                    0,
                    1,
                ],
            )

            fig = (
                add_entrance_animation(
                    fig,

                    steps=22,
                )
            )

            render_animated_chart(
                fig,

                height=420,

                frame_duration=35,
            )


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

with prediction_tab:

    st.write("")

    section_header(
        "Prediction Studio",

        "Customer subscription prediction",

        (
            "Enter customer characteristics "
            "and estimate subscription probability."
        ),
    )

    prediction_model = (
        st.selectbox(
            "Prediction model",

            list(
                available_models()
                .keys()
            ),

            key=
                "prediction_model",
        )
    )

    result = (
        cached_model(
            prediction_model
        )
    )

    pipeline = (
        result[
            "pipeline"
        ]
    )

    form_values = {}


    with st.form(
        "prediction_form"
    ):

        (
            left,
            right,
        ) = (
            st.columns(
                2
            )
        )

        for (
            index,
            feature,
        ) in enumerate(
            X.columns
        ):

            target_column = (
                left

                if (
                    index % 2
                    == 0
                )

                else right
            )

            label = (
                feature
                .replace(
                    "_",
                    " "
                )
                .title()
            )

            with target_column:

                if (
                    feature
                    in categorical_features
                ):

                    options = (
                        sorted(
                            X[
                                feature
                            ]

                            .astype(
                                str
                            )

                            .dropna()

                            .unique()

                            .tolist()
                        )
                    )

                    form_values[
                        feature
                    ] = (
                        st.selectbox(
                            label,

                            options,

                            key=
                                f"predict_{feature}",
                        )
                    )

                else:

                    form_values[
                        feature
                    ] = (
                        st.number_input(
                            label,

                            value=float(
                                X[
                                    feature
                                ].median()
                            ),

                            key=
                                f"predict_{feature}",
                        )
                    )

        submitted = (
            st.form_submit_button(
                "Generate prediction"
            )
        )


    if submitted:

        customer = (
            pd.DataFrame(
                [
                    form_values
                ]
            )
        )

        probability = (
            pipeline
            .predict_proba(
                customer
            )[
                0,
                1,
            ]
        )

        st.write("")

        (
            probability_col,
            decision_col,
        ) = (
            st.columns(
                [
                    1,
                    1.5,
                ]
            )
        )


        # ====================================================
        # PROBABILITY
        # ====================================================

        with probability_col:

            with st.container(
                border=True
            ):

                section_header(
                    "Probability",

                    "Subscription likelihood",
                )

                active_color = (
                    GREEN

                    if (
                        probability
                        >= .5
                    )

                    else BLUE
                )

                fig = (
                    go.Figure(
                        go.Pie(
                            values=[
                                probability,

                                1
                                - probability,
                            ],

                            labels=[
                                "Subscription",

                                "Remaining",
                            ],

                            hole=.72,

                            marker=dict(
                                colors=[
                                    active_color,

                                    "#E8EDF5",
                                ],

                                line=dict(
                                    color=
                                        "white",

                                    width=5,
                                ),
                            ),

                            sort=False,

                            textinfo=
                                "none",

                            hovertemplate=(
                                "<b>%{label}</b><br>"
                                "%{percent}"
                                "<extra></extra>"
                            ),
                        )
                    )
                )

                fig.update_layout(
                    height=295,

                    paper_bgcolor=
                        "rgba(0,0,0,0)",

                    plot_bgcolor=
                        "rgba(0,0,0,0)",

                    showlegend=False,

                    margin=dict(
                        l=10,
                        r=10,
                        t=5,
                        b=5,
                    ),

                    annotations=[
                        dict(
                            x=.5,
                            y=.5,

                            text=(
                                f"<b>{probability:.1%}</b>"

                                "<br>"

                                "<span "
                                "style='"
                                "font-size:10px;"
                                "color:#8792A8'>"

                                "Probability"

                                "</span>"
                            ),

                            showarrow=False,

                            font=dict(
                                color=
                                    TEXT,

                                size=24,
                            ),
                        )
                    ],
                )

                st.plotly_chart(
                    fig,

                    use_container_width=True,

                    config={
                        "displayModeBar":
                            False,

                        "responsive":
                            True,
                    },

                    key=
                        "prediction_donut",
                )


        # ====================================================
        # DECISION
        # ====================================================

        with decision_col:

            with st.container(
                border=True
            ):

                section_header(
                    "Decision",

                    "Prediction interpretation",
                )

                if (
                    probability
                    >= .5
                ):

                    st.success(
                        """
                        **Likely to subscribe**

                        The estimated probability is above
                        the default 50% classification threshold.
                        """
                    )

                else:

                    st.info(
                        """
                        **Unlikely to subscribe**

                        The estimated probability is below
                        the default 50% classification threshold.
                        """
                    )

                (
                    metric1,
                    metric2,
                ) = (
                    st.columns(
                        2
                    )
                )

                metric1.metric(
                    "Model",

                    prediction_model,
                )

                metric2.metric(
                    "Threshold",

                    "50%",
                )

                st.caption(
                    """
                    The threshold can be adjusted depending
                    on campaign cost, expected customer value,
                    and the cost of false-positive or
                    false-negative predictions.
                    """
                )