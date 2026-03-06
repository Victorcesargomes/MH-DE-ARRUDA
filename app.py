from __future__ import annotations

import logging
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

########################
# Configurações gerais #
########################
BASE_PATH = Path(__file__).parent
CSV_PATH = BASE_PATH / "dados.csv"
CERT_PATH = BASE_PATH / "certidoes"
LOGO_PATH = BASE_PATH / "logoo.png"
CLIENT_NAME = "MH de Arruda"
MODEL_NAME = "llama-3.3-70b-versatile"

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
CONTABIL_API_URL = os.getenv("CONTABIL_API_URL")

#########################
# Configuração do logger #
#########################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.FileHandler(BASE_PATH / "app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ─── Tema Premium Dark ───────────────────────────────────────────────────────
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --bg-base:        #0a0d14;
    --bg-surface:     #10141f;
    --bg-card:        #141929;
    --bg-card-hover:  #1a2035;
    --border:         #1e2840;
    --border-light:   #253050;
    --accent-green:   #00e5a0;
    --accent-blue:    #3b82f6;
    --accent-red:     #f43f5e;
    --accent-amber:   #f59e0b;
    --text-primary:   #e8edf5;
    --text-secondary: #7a8aaa;
    --text-muted:     #3d4f6e;
    --radius:         14px;
    --shadow:         0 8px 32px rgba(0,0,0,0.45);
    --glow-green:     0 0 28px rgba(0,229,160,0.15);
    --glow-blue:      0 0 28px rgba(59,130,246,0.15);
}

/* ─── Base ─── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-base) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ─── Header ─── */
.dash-header {
    background: linear-gradient(135deg, #0d1526 0%, #111827 60%, #0d1a2e 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.04);
    position: relative;
    overflow: hidden;
}
.dash-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,229,160,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.dash-header-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 0;
}
.dash-header-sub {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-top: 4px;
    font-weight: 300;
}
.dash-header-badge {
    background: rgba(0,229,160,0.1);
    border: 1px solid rgba(0,229,160,0.25);
    color: var(--accent-green);
    font-size: 0.72rem;
    font-weight: 500;
    padding: 5px 14px;
    border-radius: 100px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ─── Section Label ─── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 12px;
    padding-left: 2px;
}

/* ─── KPI Cards ─── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
    cursor: default;
}
.kpi-card:hover {
    border-color: var(--border-light);
    transform: translateY(-2px);
}
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-card.green::after { background: linear-gradient(90deg, var(--accent-green), transparent); }
.kpi-card.blue::after  { background: linear-gradient(90deg, var(--accent-blue),  transparent); }
.kpi-card.red::after   { background: linear-gradient(90deg, var(--accent-red),   transparent); }

.kpi-icon {
    font-size: 1.3rem;
    margin-bottom: 12px;
    display: block;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--text-primary);
    line-height: 1;
}
.kpi-card.green .kpi-value { color: var(--accent-green); }
.kpi-card.red   .kpi-value { color: var(--accent-red);   }

.kpi-delta {
    font-size: 0.75rem;
    margin-top: 8px;
    color: var(--text-secondary);
    font-weight: 300;
}
.kpi-delta .up   { color: var(--accent-green); }
.kpi-delta .down { color: var(--accent-red);   }

/* ─── Chart Containers ─── */
.chart-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: var(--shadow);
}
.chart-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
}
.chart-subtitle {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-bottom: 18px;
}

/* ─── Comparativo ─── */
.compare-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}
.compare-period {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
}
.compare-period-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 10px;
}
.compare-row {
    display: flex;
    justify-content: space-between;
    padding: 7px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.83rem;
}
.compare-row:last-child { border-bottom: none; }
.compare-row-label { color: var(--text-secondary); }
.compare-row-value { font-weight: 500; color: var(--text-primary); }
.compare-row-value.up   { color: var(--accent-green); }
.compare-row-value.down { color: var(--accent-red);   }

/* ─── Date Filter ─── */
.filter-bar {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 22px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.filter-bar-icon {
    font-size: 1rem;
    color: var(--accent-blue);
}

/* ─── Chat Styling ─── */
.chat-wrapper {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-top: 8px;
}
[data-testid="stChatMessage"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 14px 0 !important;
}
[data-testid="stChatMessage"]:last-child { border-bottom: none !important; }

[data-testid="stChatInputContainer"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInputContainer"] textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ─── Streamlit overrides ─── */
h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text-primary) !important;
}
[data-testid="stMetricValue"]  { color: var(--text-primary) !important; font-family: 'Syne', sans-serif !important; }
[data-testid="stMetricLabel"]  { color: var(--text-secondary) !important; }
[data-testid="stDateInput"] input,
[data-testid="stSelectbox"]    { background: var(--bg-surface) !important; color: var(--text-primary) !important; border-color: var(--border) !important; }

div[data-testid="stTabs"] [data-testid="stTab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}
div[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent-green) !important;
    border-bottom: 2px solid var(--accent-green) !important;
    background: transparent !important;
}

.stButton > button {
    background: rgba(59,130,246,0.12) !important;
    border: 1px solid rgba(59,130,246,0.35) !important;
    color: var(--accent-blue) !important;
    border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.18s !important;
}
.stButton > button:hover {
    background: rgba(59,130,246,0.22) !important;
    border-color: var(--accent-blue) !important;
}

.stDownloadButton > button {
    background: rgba(0,229,160,0.1) !important;
    border: 1px solid rgba(0,229,160,0.3) !important;
    color: var(--accent-green) !important;
    border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide default streamlit header decoration */
[data-testid="stDecoration"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden !important; }

/* Divider */
hr { border-color: var(--border) !important; margin: 28px 0 !important; }

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: rgba(30,40,64,0.8) !important;
    border-color: var(--border-light) !important;
    color: var(--text-secondary) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    color: var(--text-primary) !important;
    border-color: var(--accent-blue) !important;
    background: rgba(59,130,246,0.1) !important;
}

/* Success/Warning/Info boxes */
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border-color: var(--border-light) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 10px; }
</style>
"""

# ─── Plotly dark layout padrão ───────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#7a8aaa", size=12),
    xaxis=dict(gridcolor="#1e2840", showgrid=True, zeroline=False,
               tickfont=dict(color="#7a8aaa")),
    yaxis=dict(gridcolor="#1e2840", showgrid=True, zeroline=False,
               tickfont=dict(color="#7a8aaa")),
    margin=dict(t=20, b=40, l=10, r=10),
)

######################################
# Index simples das certidões locais #
######################################

def indexar_certidoes(pasta: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if pasta.exists():
        for arquivo in pasta.glob("*.pdf"):
            slug = (
                arquivo.stem.lower()
                .replace("_", " ")
                .replace("-", " ")
                .replace("  ", " ")
                .strip()
            )
            slug_ascii = (
                unicodedata.normalize("NFKD", slug)
                .encode("ascii", "ignore")
                .decode("ascii")
            )
            index[slug_ascii] = arquivo
    return index


CERTIDOES = indexar_certidoes(CERT_PATH)
logger.info("Certidões indexadas: %s", list(CERTIDOES.keys()))

###############################
# Utilitários de normalização #
###############################

def normalizar_txt(txt: str) -> str:
    ascii_txt = (
        unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
    )
    ascii_txt = re.sub(r"[^a-z0-9 ]", " ", ascii_txt.lower())
    ascii_txt = re.sub(r"\s+", " ", ascii_txt).strip()
    return ascii_txt

###############################
# Funções utilitárias do CSV  #
###############################

@st.cache_data(show_spinner=False)
def carregar_df(caminho: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(caminho, sep=";", dtype=str)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        def _parse_money(txt: str) -> float:
            if pd.isna(txt) or str(txt).strip() == "":
                return 0.0
            txt = str(txt).strip().replace("\u00A0", "")
            txt = txt.replace(".", "").replace(",", ".")
            try:
                return float(txt)
            except ValueError:
                return 0.0

        for col in ("faturamento", "despesa", "lucro"):
            if col in df.columns:
                df[col] = df[col].apply(_parse_money)
            else:
                df[col] = 0.0

        if "data" in df.columns:
            df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")

        return df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as exc:
        logger.exception("Erro ao carregar CSV: %s", exc)
        return pd.DataFrame()


def filtrar_por_periodo(df: pd.DataFrame, data_inicio: datetime, data_fim: datetime) -> pd.DataFrame:
    if df.empty or "data" not in df.columns:
        return df
    return df[
        (df["data"] >= pd.Timestamp(data_inicio))
        & (df["data"] <= pd.Timestamp(data_fim))
    ]


def df_para_prompt(df: pd.DataFrame) -> str:
    if df.empty:
        return "Nenhum dado disponível."
    df_display = df.copy()
    for col in ("faturamento", "despesa", "lucro"):
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            )
    MAX_LINHAS = 200
    if len(df_display) > MAX_LINHAS:
        df_display = df_display.tail(MAX_LINHAS)
    return df_display.to_csv(index=False, sep=";")


dados_df_completo = carregar_df(CSV_PATH)

##############################
# Análise Financeira          #
##############################

def analisar_financas(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"top_despesas": pd.DataFrame(), "faturamento_medio_diario": 0.0,
                "margem_lucro": 0.0, "despesas_recorrentes": pd.Series()}
    top_despesas = df.nlargest(5, "despesa")[["descricao", "despesa"]] if "descricao" in df.columns else pd.DataFrame()
    fat_medio = df[df["faturamento"] > 0]["faturamento"].mean() if not df.empty else 0.0
    margem = (df["lucro"].sum() / df["faturamento"].sum() * 100) if df["faturamento"].sum() > 0 else 0.0
    desp_rec = (df[df["despesa"] > 0].groupby("descricao")["despesa"].sum().nlargest(5)
                if "descricao" in df.columns else pd.Series())
    return {"top_despesas": top_despesas, "faturamento_medio_diario": fat_medio,
            "margem_lucro": margem, "despesas_recorrentes": desp_rec}


def comparar_periodos(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Dict[str, Any]:
    def totais(df):
        return {
            "faturamento": df["faturamento"].sum() if not df.empty else 0.0,
            "despesa":     df["despesa"].sum()     if not df.empty else 0.0,
            "lucro":       df["lucro"].sum()        if not df.empty else 0.0,
        }
    a, b = totais(df_a), totais(df_b)
    delta = {}
    for k in a:
        delta[k] = ((a[k] - b[k]) / b[k] * 100) if b[k] != 0 else 0.0
    return {"periodo_a": a, "periodo_b": b, "delta": delta}

############################
# Gráficos Sofisticados     #
############################

def _fmt_brl(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def plot_despesas_donut(df: pd.DataFrame):
    if df.empty or "descricao" not in df.columns:
        return None
    agrup = df[df["despesa"] > 0].groupby("descricao")["despesa"].sum().nlargest(10)
    if agrup.empty:
        return None
    d = agrup.reset_index()
    d.columns = ["descricao", "valor"]
    COLORS = ["#00e5a0","#3b82f6","#f59e0b","#f43f5e","#a78bfa",
              "#22d3ee","#fb923c","#4ade80","#e879f9","#94a3b8"]
    fig = go.Figure(go.Pie(
        labels=d["descricao"], values=d["valor"],
        hole=0.62,
        marker=dict(colors=COLORS[:len(d)], line=dict(color="#0a0d14", width=2)),
        textposition="outside",
        textfont=dict(size=11, color="#7a8aaa"),
        hovertemplate="<b>%{label}</b><br>%{customdata}<br>%{percent}<extra></extra>",
        customdata=[_fmt_brl(v) for v in d["valor"]],
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5,
                    bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                    font=dict(size=11, color="#7a8aaa")),
        annotations=[dict(text="Despesas", x=0.5, y=0.5, font_size=13,
                         font_color="#e8edf5", font_family="Syne, sans-serif",
                         showarrow=False)],
        height=360,
    )
    return fig


def plot_evolucao_area(df: pd.DataFrame):
    if df.empty or "data" not in df.columns:
        return None
    ag = df.groupby("data")[["faturamento", "despesa", "lucro"]].sum().reset_index()
    if ag.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ag["data"], y=ag["faturamento"], name="Faturamento",
        fill="tozeroy", mode="lines",
        line=dict(color="#00e5a0", width=2),
        fillcolor="rgba(0,229,160,0.07)",
        hovertemplate="<b>%{x|%d/%m/%y}</b><br>Faturamento: R$ %{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ag["data"], y=ag["despesa"], name="Despesa",
        fill="tozeroy", mode="lines",
        line=dict(color="#f43f5e", width=2),
        fillcolor="rgba(244,63,94,0.07)",
        hovertemplate="<b>%{x|%d/%m/%y}</b><br>Despesa: R$ %{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ag["data"], y=ag["lucro"], name="Lucro",
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2.5, dash="dot"),
        marker=dict(size=5, color="#3b82f6"),
        hovertemplate="<b>%{x|%d/%m/%y}</b><br>Lucro: R$ %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        hovermode="x unified",
        height=340,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                    font=dict(size=11, color="#7a8aaa")),
    )
    return fig


def plot_barras_mensais(df: pd.DataFrame):
    if df.empty or "data" not in df.columns:
        return None
    df2 = df.copy()
    df2["mes"] = df2["data"].dt.to_period("M").astype(str)
    ag = df2.groupby("mes")[["faturamento", "despesa", "lucro"]].sum().reset_index()
    if len(ag) < 2:
        return None

    # customdata com os 3 valores formatados por linha do agrupado
    import numpy as np
    cd = np.stack([
        ag["faturamento"].values,
        ag["despesa"].values,
        ag["lucro"].values,
    ], axis=1)  # shape (n_meses, 3)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ag["mes"], y=ag["faturamento"], name="Faturamento",
        marker_color="rgba(0,229,160,0.75)",
        customdata=cd,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Faturamento: R$ %{customdata[0]:,.2f}<br>"
            "Despesa:     R$ %{customdata[1]:,.2f}<br>"
            "Lucro:       R$ %{customdata[2]:,.2f}"
            "<extra></extra>"
        ),
    ))
    fig.add_trace(go.Bar(
        x=ag["mes"], y=ag["despesa"], name="Despesa",
        marker_color="rgba(244,63,94,0.65)",
        customdata=cd,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Faturamento: R$ %{customdata[0]:,.2f}<br>"
            "Despesa:     R$ %{customdata[1]:,.2f}<br>"
            "Lucro:       R$ %{customdata[2]:,.2f}"
            "<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=ag["mes"], y=ag["lucro"], name="Lucro",
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=6),
        yaxis="y2",
        customdata=cd,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Faturamento: R$ %{customdata[0]:,.2f}<br>"
            "Despesa:     R$ %{customdata[1]:,.2f}<br>"
            "Lucro:       R$ %{customdata[2]:,.2f}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        barmode="group",
        height=330,
        hovermode="closest",
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    tickfont=dict(color="#3b82f6"), zeroline=False),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                    font=dict(size=11, color="#7a8aaa")),
    )
    return fig


def plot_waterfall_lucro(df: pd.DataFrame):
    if df.empty:
        return None
    fat = df["faturamento"].sum()
    desp = df["despesa"].sum()
    lucro = df["lucro"].sum()
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["Faturamento", "Despesas", "Lucro Líquido"],
        y=[fat, -desp, lucro],
        connector=dict(line=dict(color="#1e2840", width=1.5)),
        increasing=dict(marker_color="#00e5a0"),
        decreasing=dict(marker_color="#f43f5e"),
        totals=dict(marker_color="#3b82f6"),
        text=[_fmt_brl(fat), f"−{_fmt_brl(desp)}", _fmt_brl(lucro)],
        textposition="outside",
        textfont=dict(color="#e8edf5", size=11),
        hovertemplate="<b>%{x}</b><br>R$ %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(**PLOT_LAYOUT, height=300,
                      legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"))
    return fig


def plot_comparativo_barras(comp: Dict):
    a = comp["periodo_a"]
    b = comp["periodo_b"]
    cats = ["Faturamento", "Despesa", "Lucro"]
    vals_a = [a["faturamento"], a["despesa"], a["lucro"]]
    vals_b = [b["faturamento"], b["despesa"], b["lucro"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Período A", x=cats, y=vals_a,
        marker_color="rgba(59,130,246,0.75)",
        hovertemplate="%{x}: R$ %{y:,.2f}<extra>Período A</extra>",
    ))
    fig.add_trace(go.Bar(
        name="Período B", x=cats, y=vals_b,
        marker_color="rgba(0,229,160,0.6)",
        hovertemplate="%{x}: R$ %{y:,.2f}<extra>Período B</extra>",
    ))
    fig.update_layout(**PLOT_LAYOUT, barmode="group", height=290,
                      legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center",
                                  bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                                  font=dict(size=11, color="#7a8aaa")))
    return fig

###################################
# Histórico de mensagens          #
###################################

def get_historico() -> List[BaseMessage]:
    return st.session_state.setdefault("historico", [])


def adicionar_mensagem(role: str, conteudo: str) -> None:
    historico = get_historico()
    msg = HumanMessage(content=conteudo) if role == "human" else AIMessage(content=conteudo)
    historico.append(msg)
    if len(historico) > 40:
        st.session_state["historico"] = historico[-40:]


def limpar_historico() -> None:
    st.session_state["historico"] = []

###################################
# Configuração do modelo (Groq)   #
###################################

@st.cache_resource
def _criar_client() -> ChatGroq:
    return ChatGroq(api_key=API_KEY, model=MODEL_NAME)


def _criar_chain(llm: ChatGroq, df_filtrado: pd.DataFrame, data_inicio, data_fim):
    faturamento = df_filtrado["faturamento"].sum() if not df_filtrado.empty and "faturamento" in df_filtrado.columns else 0.0
    despesa     = df_filtrado["despesa"].sum()     if not df_filtrado.empty and "despesa" in df_filtrado.columns else 0.0
    lucro       = df_filtrado["lucro"].sum()       if not df_filtrado.empty and "lucro" in df_filtrado.columns else 0.0
    analise     = analisar_financas(df_filtrado)
    periodo_str = f"{data_inicio.strftime('%d/%m/%Y')} a {data_fim.strftime('%d/%m/%Y')}"

    system_message = f"""
Você é Victor, assistente virtual da empresa de Reciclagem \"{CLIENT_NAME}\" de Matheus e Márcia.
Fale **sempre** em português brasileiro, de forma clara e objetiva.
Ignore qualquer texto entre as tags <think> e </think>; trate-o como nota interna.
Você interage exclusivamente com Matheus ou Márcia, donos da empresa MH de Arruda.
Antes de iniciar a conversa, pergunte com quem está falando.

**DIRETRIZES DE RESPOSTA:**
1. Seja conciso. Evite textos longos quando a pergunta for simples.
2. Para perguntas objetivas, responda em no máximo 2 frases.
3. Para perguntas sobre valores financeiros, mostre apenas os números relevantes.
4. Para pedidos de certidões, forneça imediatamente o link de download.
5. Use negrito apenas para valores numéricos importantes.

**Resumo financeiro do período {periodo_str}**:
- **Faturamento acumulado:** R$ {faturamento:,.2f}
- **Despesa acumulada:** R$ {despesa:,.2f}
- **Lucro acumulado:** R$ {lucro:,.2f}

**Análise detalhada**:
- **Top 5 despesas**:
{analise['top_despesas'].to_string(index=False) if not analise['top_despesas'].empty else 'Nenhuma despesa registrada'}
- **Faturamento médio diário**: R$ {analise['faturamento_medio_diario']:,.2f}
- **Margem de lucro**: {analise['margem_lucro']:.2f}%
- **Principais despesas recorrentes**:
{analise['despesas_recorrentes'].to_string() if not analise['despesas_recorrentes'].empty else 'Nenhuma despesa recorrente'}

Certidões disponíveis: {', '.join(CERTIDOES.keys()) or 'nenhuma'}

Base de dados detalhada (período {periodo_str}):
###
{df_para_prompt(df_filtrado)}
###

- As colunas correspondem a `Data`, `Faturamento`, `Despesa`, `Descrição` e `Lucro`.
- Todos os valores estão em Reais (BRL).

Se precisar de uma certidão, basta pedir — exemplos: "quero a CND estadual", "quero a CND FGTS".
Termine sempre perguntando se precisa de algo mais.
"""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ])
    return prompt_template | llm

#######################################
# Certidões PDF                        #
#######################################

CATEGORIAS_CERT = {
    "estadual": "Estadual", "federal": "Federal",
    "municipal": "Municipal", "fgts": "FGTS", "fiscal": "Fiscal",
}


def tentar_enviar_certidao(mensagem: str) -> Tuple[Optional[Path], Optional[str]]:
    txt = normalizar_txt(mensagem)
    if "cnd" not in txt and "certidao" not in txt:
        return None, None
    for cat_slug, cat_nome in CATEGORIAS_CERT.items():
        if cat_slug in txt:
            for slug, path in CERTIDOES.items():
                if cat_slug in slug and ("cnd" in slug or "certidao" in slug):
                    return path, cat_nome
    return None, None

################################
# Contabilidade                 #
################################

def enviar_contabilidade(df: pd.DataFrame) -> bool:
    if not CONTABIL_API_URL:
        return False
    try:
        dados_json = df.assign(
            data=df["data"].dt.strftime("%Y-%m-%d").fillna("")
        ).to_dict(orient="records")
        response = requests.post(CONTABIL_API_URL, json=dados_json, timeout=10)
        return response.status_code == 200
    except Exception as exc:
        logger.exception("Erro na integração contábil: %s", exc)
        return False

###########################
# LLM                      #
###########################

def consultar_modelo(chain, entrada: str) -> str:
    try:
        resposta = chain.invoke({
            "input": entrada,
            "chat_history": get_historico(),
        }).content
        return resposta
    except Exception as exc:
        logger.exception("Erro na chamada do LLM: %s", exc)
        return "❌ Ocorreu um erro. Tente novamente."

###########################
# Helpers de UI            #
###########################

def fmt_brl(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def delta_html(pct: float) -> str:
    if pct > 0:
        return f'<span class="up">▲ {pct:.1f}%</span>'
    elif pct < 0:
        return f'<span class="down">▼ {abs(pct):.1f}%</span>'
    return f'<span>— 0%</span>'

###########################
# Sidebar                   #
###########################

def desenhar_sidebar(llm: ChatGroq) -> None:
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=260)
        else:
            st.markdown(
                f'<div style="font-family:Syne,sans-serif;font-size:1.2rem;'
                f'font-weight:800;color:#e8edf5;padding:16px 0 8px;">{CLIENT_NAME}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Navegação</div>', unsafe_allow_html=True)
        abas = st.tabs(["💬 Chat", "⚙️ Config"])

        with abas[0]:
            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
            if st.button("🗑️ Limpar Histórico", use_container_width=True):
                limpar_historico()
                st.success("Histórico limpo.")
            st.markdown(
                f'<div style="margin-top:20px;font-size:0.75rem;color:#3d4f6e;line-height:1.7">'
                f'Modelo: <span style="color:#7a8aaa">{MODEL_NAME}</span><br>'
                f'Linhas CSV: <span style="color:#7a8aaa">{len(dados_df_completo)}</span><br>'
                f'Certidões: <span style="color:#7a8aaa">{len(CERTIDOES)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with abas[1]:
            st.markdown('<div class="section-label" style="margin-top:12px">Integração</div>',
                        unsafe_allow_html=True)
            if CONTABIL_API_URL:
                if st.button("📤 Enviar para Contabilidade", use_container_width=True):
                    ok = enviar_contabilidade(dados_df_completo)
                    st.success("Enviado!") if ok else st.error("Falha no envio.")
            else:
                st.caption("Integração contábil não configurada.")

            st.markdown('<div class="section-label" style="margin-top:20px">Certidões</div>',
                        unsafe_allow_html=True)
            if CERTIDOES:
                for nome, path in CERTIDOES.items():
                    with open(path, "rb") as f:
                        st.download_button(
                            label=f"📄 {nome.title()}",
                            data=f.read(),
                            file_name=path.name,
                            mime="application/pdf",
                            use_container_width=True,
                        )
            else:
                st.caption("Nenhuma certidão disponível.")

###########################
# Página Principal         #
###########################

def pagina_chat(llm: ChatGroq) -> None:
    # ─── Header ───────────────────────────────────────────────────────────────
    hoje = datetime.now().strftime("%d/%m/%Y")
    st.markdown(
        f"""
        <div class="dash-header">
            <div>
                <div class="dash-header-title">📊 Dashboard Financeiro</div>
                <div class="dash-header-sub">{CLIENT_NAME} · Atualizado em {hoje}</div>
            </div>
            <div class="dash-header-badge">● Online</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Validação de datas ───────────────────────────────────────────────────
    datas_validas = (
        dados_df_completo["data"].dropna()
        if not dados_df_completo.empty and "data" in dados_df_completo.columns
        else pd.Series()
    )
    if datas_validas.empty:
        st.warning("Nenhuma data válida no CSV.")
        return

    min_date = datas_validas.min().to_pydatetime().date()
    max_date = datas_validas.max().to_pydatetime().date()

    # ─── Abas principais ──────────────────────────────────────────────────────
    tab_visao, tab_comparativo, tab_chat = st.tabs([
        "  Visão Geral  ", "  Comparativo  ", "  Analista IA  "
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — VISÃO GERAL
    # ══════════════════════════════════════════════════════════════════════════
    with tab_visao:
        st.markdown('<div class="section-label" style="margin-top:16px">Filtro de Período</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        data_inicio = c1.date_input("De", min_date, min_value=min_date,
                                    max_value=max_date, key="vi_ini")
        data_fim    = c2.date_input("Até", max_date, min_value=min_date,
                                    max_value=max_date, key="vi_fim")

        if data_inicio > data_fim:
            st.warning("Data inicial posterior à data final.")
            return

        dt_ini = datetime.combine(data_inicio, datetime.min.time())
        dt_fim = datetime.combine(data_fim, datetime.max.time())
        dados_df = filtrar_por_periodo(dados_df_completo, dt_ini, dt_fim)

        if dados_df.empty:
            st.info("Nenhum dado no período.")
            return

        fat = dados_df["faturamento"].sum()
        desp = dados_df["despesa"].sum()
        lucro = dados_df["lucro"].sum()
        margem = (lucro / fat * 100) if fat > 0 else 0.0

        # ─── KPI Cards ────────────────────────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top:20px">Indicadores</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card green">
                    <span class="kpi-icon">💰</span>
                    <div class="kpi-label">Faturamento Total</div>
                    <div class="kpi-value">{fmt_brl(fat)}</div>
                    <div class="kpi-delta">Período selecionado</div>
                </div>
                <div class="kpi-card red">
                    <span class="kpi-icon">📉</span>
                    <div class="kpi-label">Despesa Total</div>
                    <div class="kpi-value">{fmt_brl(desp)}</div>
                    <div class="kpi-delta">Período selecionado</div>
                </div>
                <div class="kpi-card blue">
                    <span class="kpi-icon">📈</span>
                    <div class="kpi-label">Lucro Líquido</div>
                    <div class="kpi-value">{fmt_brl(lucro)}</div>
                    <div class="kpi-delta">Margem: <b>{margem:.1f}%</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ─── Waterfall ────────────────────────────────────────────────────────
        st.markdown('<div class="chart-card">'
                    '<div class="chart-title">Resumo Financeiro</div>'
                    '<div class="chart-subtitle">Fluxo de Faturamento → Lucro Líquido</div>',
                    unsafe_allow_html=True)
        fig_wf = plot_waterfall_lucro(dados_df)
        if fig_wf:
            st.plotly_chart(fig_wf, width="stretch", config={"displayModeBar": False}, key="chart_waterfall")
        st.markdown('</div>', unsafe_allow_html=True)

        # ─── Linha e Donut ────────────────────────────────────────────────────
        col_g1, col_g2 = st.columns([3, 2])

        with col_g1:
            st.markdown('<div class="chart-card">'
                        '<div class="chart-title">Evolução no Período</div>'
                        '<div class="chart-subtitle">Faturamento · Despesa · Lucro ao longo do tempo</div>',
                        unsafe_allow_html=True)
            fig_ev = plot_evolucao_area(dados_df)
            if fig_ev:
                st.plotly_chart(fig_ev, width="stretch", config={"displayModeBar": False}, key="chart_evolucao")
            else:
                st.info("Dados insuficientes.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_g2:
            st.markdown('<div class="chart-card">'
                        '<div class="chart-title">Top 10 Despesas</div>'
                        '<div class="chart-subtitle">Distribuição por categoria</div>',
                        unsafe_allow_html=True)
            fig_do = plot_despesas_donut(dados_df)
            if fig_do:
                st.plotly_chart(fig_do, width="stretch", config={"displayModeBar": False}, key="chart_donut")
            else:
                st.info("Sem despesas.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ─── Barras mensais ───────────────────────────────────────────────────
        st.markdown('<div class="chart-card">'
                    '<div class="chart-title">Consolidado Mensal</div>'
                    '<div class="chart-subtitle">Barras agrupadas + linha de lucro (eixo secundário)</div>',
                    unsafe_allow_html=True)
        fig_bar = plot_barras_mensais(dados_df)
        if fig_bar:
            st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False}, key="chart_barras_mensais")
        else:
            st.info("São necessários pelo menos 2 meses de dados.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — COMPARATIVO
    # ══════════════════════════════════════════════════════════════════════════
    with tab_comparativo:
        st.markdown('<div class="section-label" style="margin-top:16px">Comparativo Entre Períodos</div>',
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Período A**")
            ini_a = st.date_input("Início A", min_date, min_value=min_date,
                                  max_value=max_date, key="ca_ini")
            fim_a = st.date_input("Fim A",    max_date, min_value=min_date,
                                  max_value=max_date, key="ca_fim")
        with c2:
            st.markdown("**Período B**")
            ini_b = st.date_input("Início B", min_date, min_value=min_date,
                                  max_value=max_date, key="cb_ini")
            fim_b = st.date_input("Fim B",    max_date, min_value=min_date,
                                  max_value=max_date, key="cb_fim")

        df_a = filtrar_por_periodo(dados_df_completo,
                                   datetime.combine(ini_a, datetime.min.time()),
                                   datetime.combine(fim_a, datetime.max.time()))
        df_b = filtrar_por_periodo(dados_df_completo,
                                   datetime.combine(ini_b, datetime.min.time()),
                                   datetime.combine(fim_b, datetime.max.time()))

        comp = comparar_periodos(df_a, df_b)
        a, b, delta = comp["periodo_a"], comp["periodo_b"], comp["delta"]

        # Cards comparativos
        st.markdown(
            f"""
            <div class="compare-grid" style="margin-top:20px">
                <div class="compare-period">
                    <div class="compare-period-label">📅 Período A &nbsp;|&nbsp;
                        {ini_a.strftime("%d/%m/%y")} – {fim_a.strftime("%d/%m/%y")}</div>
                    <div class="compare-row">
                        <span class="compare-row-label">Faturamento</span>
                        <span class="compare-row-value">{fmt_brl(a["faturamento"])}</span>
                    </div>
                    <div class="compare-row">
                        <span class="compare-row-label">Despesa</span>
                        <span class="compare-row-value">{fmt_brl(a["despesa"])}</span>
                    </div>
                    <div class="compare-row">
                        <span class="compare-row-label">Lucro</span>
                        <span class="compare-row-value {'up' if a['lucro']>=0 else 'down'}">{fmt_brl(a["lucro"])}</span>
                    </div>
                </div>
                <div class="compare-period">
                    <div class="compare-period-label">📅 Período B &nbsp;|&nbsp;
                        {ini_b.strftime("%d/%m/%y")} – {fim_b.strftime("%d/%m/%y")}</div>
                    <div class="compare-row">
                        <span class="compare-row-label">Faturamento</span>
                        <span class="compare-row-value">{fmt_brl(b["faturamento"])}</span>
                    </div>
                    <div class="compare-row">
                        <span class="compare-row-label">Despesa</span>
                        <span class="compare-row-value">{fmt_brl(b["despesa"])}</span>
                    </div>
                    <div class="compare-row">
                        <span class="compare-row-label">Lucro</span>
                        <span class="compare-row-value {'up' if b['lucro']>=0 else 'down'}">{fmt_brl(b["lucro"])}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Variação A vs B
        st.markdown(
            f"""
            <div class="chart-card" style="margin-top:20px">
                <div class="chart-title">Variação A vs B</div>
                <div class="chart-subtitle">Diferença percentual do Período A em relação ao Período B</div>
                <div style="display:flex;gap:32px;margin-top:12px">
                    <div>
                        <div style="font-size:0.75rem;color:#7a8aaa;margin-bottom:4px">Faturamento</div>
                        <div style="font-size:1.3rem;font-family:Syne,sans-serif;font-weight:700">
                            {delta_html(delta["faturamento"])}
                        </div>
                    </div>
                    <div>
                        <div style="font-size:0.75rem;color:#7a8aaa;margin-bottom:4px">Despesa</div>
                        <div style="font-size:1.3rem;font-family:Syne,sans-serif;font-weight:700">
                            {delta_html(-delta["despesa"])}
                        </div>
                    </div>
                    <div>
                        <div style="font-size:0.75rem;color:#7a8aaa;margin-bottom:4px">Lucro</div>
                        <div style="font-size:1.3rem;font-family:Syne,sans-serif;font-weight:700">
                            {delta_html(delta["lucro"])}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="chart-card" style="margin-top:4px">'
                    '<div class="chart-title">Comparativo Visual</div>'
                    '<div class="chart-subtitle">Período A vs Período B por categoria</div>',
                    unsafe_allow_html=True)
        fig_comp = plot_comparativo_barras(comp)
        if fig_comp:
            st.plotly_chart(fig_comp, width="stretch", config={"displayModeBar": False}, key="chart_comparativo")
        st.markdown('</div>', unsafe_allow_html=True)

        # Área comparativa temporal
        if not df_a.empty and not df_b.empty:
            st.markdown('<div class="chart-card" style="margin-top:4px">'
                        '<div class="chart-title">Evolução — Período A</div>'
                        '<div class="chart-subtitle">Faturamento · Despesa · Lucro</div>',
                        unsafe_allow_html=True)
            fa = plot_evolucao_area(df_a)
            if fa:
                st.plotly_chart(fa, width="stretch", config={"displayModeBar": False}, key="chart_evolucao_a")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="chart-card" style="margin-top:4px">'
                        '<div class="chart-title">Evolução — Período B</div>'
                        '<div class="chart-subtitle">Faturamento · Despesa · Lucro</div>',
                        unsafe_allow_html=True)
            fb = plot_evolucao_area(df_b)
            if fb:
                st.plotly_chart(fb, width="stretch", config={"displayModeBar": False}, key="chart_evolucao_b")
            st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — ANALISTA IA
    # ══════════════════════════════════════════════════════════════════════════
    with tab_chat:
        st.markdown('<div class="section-label" style="margin-top:16px">Filtro de Contexto</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        chat_ini = c1.date_input("De", min_date, min_value=min_date,
                                 max_value=max_date, key="ch_ini")
        chat_fim = c2.date_input("Até", max_date, min_value=min_date,
                                 max_value=max_date, key="ch_fim")

        dados_chat = filtrar_por_periodo(
            dados_df_completo,
            datetime.combine(chat_ini, datetime.min.time()),
            datetime.combine(chat_fim, datetime.max.time()),
        )
        chain = _criar_chain(llm, dados_chat, chat_ini, chat_fim)

        # ── Download de certidão pendente (salvo antes do rerun) ──────────────
        cert_pendente = st.session_state.pop("cert_pendente", None)
        if cert_pendente:
            with open(cert_pendente["path"], "rb") as f:
                st.download_button(
                    label=f"📄 Baixar CND {cert_pendente['categoria']}",
                    data=f.read(),
                    file_name=cert_pendente["nome"],
                    mime="application/pdf",
                    key="cert_dl",
                )

        # ── Histórico de mensagens ─────────────────────────────────────────────
        historico = get_historico()
        for msg in historico:
            if isinstance(msg, HumanMessage):
                st.chat_message("human").markdown(msg.content)
            elif isinstance(msg, AIMessage):
                st.chat_message("ai").markdown(msg.content)

        # ── Input ─────────────────────────────────────────────────────────────
        entrada = st.chat_input("Pergunte sobre os dados financeiros...")
        if not entrada:
            return

        entrada_limpa = re.sub(r"<think>.*?</think>", "", entrada,
                               flags=re.DOTALL | re.IGNORECASE).strip()
        if not entrada_limpa:
            return

        # Salva mensagem do usuário e rerun — o histórico acima renderiza tudo
        adicionar_mensagem("human", entrada_limpa)

        # Certidão?
        cert_path, categoria = tentar_enviar_certidao(entrada_limpa)
        if cert_path and categoria:
            resposta = f"Aqui está a **CND {categoria}** conforme solicitado."
            adicionar_mensagem("ai", resposta)
            st.session_state["cert_pendente"] = {
                "path": str(cert_path),
                "categoria": categoria,
                "nome": cert_path.name,
            }
            st.rerun()

        # Resposta da IA — processa e salva, depois rerun para renderizar no histórico
        with st.spinner("Analisando..."):
            resposta_llm = consultar_modelo(chain, entrada_limpa)
        adicionar_mensagem("ai", resposta_llm)
        st.rerun()

###########################
# Main                     #
###########################

def main() -> None:
    st.set_page_config(
        page_title=f"Dashboard · {CLIENT_NAME}",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    if not API_KEY:
        st.error("Variável GROQ_API_KEY não encontrada. Configure o arquivo .env.")
        st.stop()

    llm = _criar_client()
    desenhar_sidebar(llm)
    pagina_chat(llm)


if __name__ == "__main__":
    main()
