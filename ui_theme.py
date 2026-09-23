"""Shared visual language for the two Streamlit workspaces."""

from __future__ import annotations

import matplotlib.pyplot as plt
from cycler import cycler
import streamlit as st


STYLES = """
<style>
  :root {
    --surface: #102233;
    --surface-soft: #142a3c;
    --border: rgba(148, 190, 207, .16);
    --text: #e8f3f4;
    --muted: #a8bdc8;
    --accent: #53ddcb;
  }

  .stApp {
    background:
      radial-gradient(circle at 83% 4%, rgba(46, 128, 142, .16), transparent 34rem),
      radial-gradient(circle at 16% 44%, rgba(51, 91, 146, .10), transparent 38rem),
      #08131f;
  }
  [data-testid="stHeader"] { background: transparent; }
  [data-testid="stAppViewContainer"] > .main .block-container {
    max-width: 1320px;
    padding: 3rem 3.5rem 5rem;
  }
  [data-testid="stSidebar"] {
    background: linear-gradient(165deg, #10283a 0%, #0b1928 62%, #0c1523 100%);
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
  [data-testid="stSidebar"] h2 {
    border: 0;
    padding: 0;
    margin-top: 1.3rem;
    font-size: .73rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .14em;
    color: #8daab9;
  }
  [data-testid="stSidebar"] [role="radiogroup"] { gap: .25rem; }
  [data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: .75rem;
    padding: .45rem .6rem;
    transition: background .15s ease;
  }
  [data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(83, 221, 203, .08);
  }
  .sidebar-brand {
    display: flex;
    align-items: center;
    gap: .8rem;
    padding: .35rem .3rem 1.2rem;
    margin-bottom: .3rem;
    border-bottom: 1px solid var(--border);
  }
  .sidebar-brand__icon {
    display: grid;
    place-items: center;
    width: 2.35rem;
    height: 2.35rem;
    border-radius: .75rem;
    background: linear-gradient(135deg, #1caeaa, #316e9a);
    color: white;
    font-size: 1.45rem;
    box-shadow: 0 7px 22px rgba(0, 193, 174, .18);
  }
  .sidebar-brand__name {
    color: var(--text);
    font-size: 1.02rem;
    font-weight: 800;
    letter-spacing: .02em;
    line-height: 1.05;
  }
  .sidebar-brand__sub {
    color: #88a6b5;
    font-size: .72rem;
    letter-spacing: .11em;
    text-transform: uppercase;
    margin-top: .3rem;
  }
  .hero {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(106, 222, 208, .2);
    border-radius: 1.25rem;
    padding: 2.2rem 2.5rem;
    margin: .25rem 0 2.1rem;
    min-height: 240px;
    background:
      radial-gradient(circle at 82% 45%, rgba(68, 206, 195, .17), transparent 18rem),
      linear-gradient(115deg, #163448, #102537 58%, #12283c);
    box-shadow: 0 18px 48px rgba(0, 5, 12, .18);
  }
  .hero--heart {
    border-color: rgba(251, 113, 133, .22);
    background:
      radial-gradient(circle at 82% 45%, rgba(251, 113, 133, .18), transparent 18rem),
      linear-gradient(115deg, #263445, #142a3b 58%, #24293c);
  }
  .hero__content { position: relative; z-index: 1; max-width: 720px; }
  .hero__eyebrow {
    color: #73e8d8;
    text-transform: uppercase;
    letter-spacing: .19em;
    font-size: .73rem;
    font-weight: 800;
    margin-bottom: .8rem;
  }
  .hero--heart .hero__eyebrow { color: #ff9dac; }
  .hero__title {
    color: #f3fbfb;
    font-weight: 800;
    font-size: clamp(2rem, 4vw, 3.35rem);
    line-height: 1.1;
    letter-spacing: -.035em;
    margin: 0 0 .8rem;
  }
  .hero__description {
    color: #c2d5db;
    font-size: 1.05rem;
    line-height: 1.55;
    max-width: 600px;
    margin: 0 0 1.25rem;
  }
  .hero__tags { display: flex; flex-wrap: wrap; gap: .5rem; }
  .hero__tag {
    border: 1px solid rgba(180, 228, 226, .19);
    border-radius: 999px;
    color: #dcf3f0;
    background: rgba(255, 255, 255, .055);
    padding: .34rem .7rem;
    font-size: .76rem;
    font-weight: 700;
  }
  .hero__art {
    position: absolute;
    right: 2.4rem;
    top: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: .28rem;
    height: 130px;
    transform: translateY(-50%);
    opacity: .72;
  }
  .hero__art span {
    width: .34rem;
    height: var(--height);
    border-radius: 99px;
    background: linear-gradient(180deg, #8bf1e0, #3b91bb);
    box-shadow: 0 0 22px rgba(83, 221, 203, .3);
  }
  .hero--heart .hero__art span {
    background: linear-gradient(180deg, #ffb2bc, #db6686);
    box-shadow: 0 0 22px rgba(251, 113, 133, .3);
  }
  .hero__art span:nth-child(1), .hero__art span:nth-child(11) { --height: 22px; }
  .hero__art span:nth-child(2), .hero__art span:nth-child(10) { --height: 45px; }
  .hero__art span:nth-child(3), .hero__art span:nth-child(9) { --height: 70px; }
  .hero__art span:nth-child(4), .hero__art span:nth-child(8) { --height: 38px; }
  .hero__art span:nth-child(5), .hero__art span:nth-child(7) { --height: 102px; }
  .hero__art span:nth-child(6) { --height: 130px; }

  [data-testid="stAppViewContainer"] .main h2 {
    color: #edf8f7 !important;
    font-size: 1.45rem !important;
    letter-spacing: -.02em;
    padding-top: 1.55rem !important;
    margin-top: 1.6rem !important;
    border-top: 1px solid var(--border);
  }
  [data-testid="stAppViewContainer"] .main h3 {
    color: #d9f0ec !important;
    font-size: 1.12rem !important;
  }
  [data-testid="stMetric"] {
    background: linear-gradient(145deg, #142a3c, #102233);
    border: 1px solid var(--border);
    border-radius: .95rem;
    padding: 1.1rem 1.3rem;
    min-height: 112px;
    box-shadow: 0 9px 24px rgba(0, 4, 12, .13);
  }
  [data-testid="stMetricLabel"] { color: #a7c0cb; }
  [data-testid="stMetricValue"] { color: #f0fbfa; font-weight: 800; }
  [data-testid="stFileUploaderDropzone"] {
    border: 1px dashed rgba(115, 232, 216, .48);
    border-radius: .95rem;
    background: rgba(25, 57, 72, .55);
  }
  [data-testid="stFileUploaderDropzone"]:hover {
    border-color: #73e8d8;
    background: rgba(32, 75, 86, .6);
  }
  [data-testid="stButton"] button,
  [data-testid="stDownloadButton"] button {
    border-radius: .7rem;
    border: 1px solid rgba(83, 221, 203, .35);
    font-weight: 700;
    transition: transform .15s ease, border-color .15s ease, background .15s ease;
  }
  [data-testid="stButton"] button:hover,
  [data-testid="stDownloadButton"] button:hover {
    transform: translateY(-1px);
    border-color: #53ddcb;
  }
  [data-testid="stImage"] img {
    border-radius: .8rem;
    border: 1px solid var(--border);
  }
  [data-testid="stAlert"] { border-radius: .8rem; }
  .empty-panel {
    border: 1px solid var(--border);
    border-radius: 1rem;
    padding: 1.7rem 2rem;
    background: linear-gradient(130deg, #142a3c, #101f2f);
    margin: 1.1rem 0 1.5rem;
  }
  .empty-panel__eyebrow {
    color: var(--accent);
    font-size: .7rem;
    font-weight: 800;
    letter-spacing: .15em;
    text-transform: uppercase;
  }
  .empty-panel h3 { font-size: 1.3rem !important; margin: .45rem 0 .3rem; }
  .empty-panel p { color: var(--muted); margin: 0; }
  @media (max-width: 950px) {
    [data-testid="stAppViewContainer"] > .main .block-container { padding: 2rem 1.25rem 3rem; }
    .hero { padding: 1.7rem; min-height: 220px; }
    .hero__art { opacity: .2; right: 1rem; }
  }
</style>
"""


def apply_theme() -> None:
    """Apply app chrome and a matching palette to Matplotlib figures."""
    st.html(STYLES)
    plt.rcParams.update({
        "figure.facecolor": "#102233",
        "axes.facecolor": "#102233",
        "savefig.facecolor": "#102233",
        "text.color": "#e8f3f4",
        "axes.labelcolor": "#c8dce2",
        "axes.edgecolor": "#577083",
        "axes.titlecolor": "#effafa",
        "xtick.color": "#9eb9c4",
        "ytick.color": "#9eb9c4",
        "grid.color": "#466173",
        "grid.alpha": 0.27,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=["#53ddcb", "#ff8398", "#94acff", "#f4c977"]),
    })


def render_sidebar_brand() -> None:
    with st.sidebar:
        st.html(
            '<div class="sidebar-brand">'
            '<div class="sidebar-brand__icon">∿</div>'
            '<div><div class="sidebar-brand__name">Signal Studio</div>'
            '<div class="sidebar-brand__sub">Audio workbench</div></div></div>'
        )


def render_hero(workspace: str) -> None:
    if workspace == "heart":
        variant = " hero--heart"
        eyebrow = "Heart sound workspace"
        title = "Explore every heartbeat."
        description = "See the waveform, isolate heart sounds, and follow the rhythm in your recording."
        tags = ("PCG analysis", "Filtering", "Beat markers")
    else:
        variant = ""
        eyebrow = "Audio analysis workspace"
        title = "Make sound visible."
        description = "Record, inspect, shape, and compare audio in one interactive workspace."
        tags = ("Live microphone", "Spectrogram", "Effects", "Export")
    bars = "".join("<span></span>" for _ in range(11))
    pills = "".join(f'<span class="hero__tag">{tag}</span>' for tag in tags)
    st.html(
        f'<div class="hero{variant}"><div class="hero__content">'
        f'<div class="hero__eyebrow">{eyebrow}</div>'
        f'<h1 class="hero__title">{title}</h1>'
        f'<p class="hero__description">{description}</p>'
        f'<div class="hero__tags">{pills}</div></div>'
        f'<div class="hero__art" aria-hidden="true">{bars}</div></div>'
    )


def render_empty_state(title: str, message: str) -> None:
    st.html(
        '<div class="empty-panel"><div class="empty-panel__eyebrow">Ready when you are</div>'
        f'<h3>{title}</h3><p>{message}</p></div>'
    )
