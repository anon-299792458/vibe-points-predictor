"""
CAO 2026 Level 8 Point Predictor — minimal single-page app.
Run: python -m streamlit run app.py
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

DB_PATH = Path(__file__).parent / "predictor.db"

st.set_page_config(page_title="CAO 2026 Vibes-Based Points Predictor", layout="centered")

st.markdown("""
<style>
  .pred-block { text-align:center; padding:18px 0 8px; }
  .pred-num   { font-size:3.2em; font-weight:800; color:#1e293b; line-height:1; }
  .pred-ci    { font-size:1.05em; color:#7c3aed; font-weight:600; margin-top:4px; }
  .trend-up   { color:#16a34a; font-weight:700; }
  .trend-down { color:#dc2626; font-weight:700; }
  .trend-stable { color:#d97706; font-weight:700; }
  .disc       { font-size:0.82em; color:#64748b; margin-top:20px;
                border-top:1px solid #e2e8f0; padding-top:12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading data …")
def load_data():
    from model import run_predictions
    records, _, _ = run_predictions()
    return records


def ensure_db():
    if DB_PATH.exists():
        return True
    st.warning("Database not found.")
    if st.button("Build database now", type="primary"):
        with st.spinner("Parsing data files (~30 s) …"):
            r = subprocess.run(
                [sys.executable, str(Path(__file__).parent / "ingest.py")],
                capture_output=True, text=True,
            )
        if r.returncode == 0:
            st.rerun()
        else:
            st.error("Ingestion failed.")
            st.code(r.stderr[-1500:])
    return False


def make_chart(rec: dict) -> go.Figure:
    years = [int(y) for y in rec["years_data"]]
    pts   = rec["points_data"]
    anom  = [bool(a) for a in rec["is_anomaly_data"]]
    pred  = rec["prediction"]

    fig = go.Figure()

    # Baseline years (solid line)
    bx = [y for y, a in zip(years, anom) if not a]
    by = [p for p, a in zip(pts,   anom) if not a]
    fig.add_trace(go.Scatter(
        x=bx, y=by, mode="lines+markers", name="Historical",
        line=dict(color="#3b82f6", width=2.5),
        marker=dict(size=8, color="#3b82f6"),
    ))

    # Grade-adjusted years 2020-2024 (orange diamonds)
    ax = [y for y, a in zip(years, anom) if a]
    ay = [p for p, a in zip(pts,   anom) if a]
    if ax:
        fig.add_trace(go.Scatter(
            x=ax, y=ay, mode="markers", name="Grade-adjusted (2020–2024)",
            marker=dict(size=11, color="#f97316", symbol="diamond",
                        line=dict(width=1.5, color="#ea580c")),
        ))

    if pred:
        pe    = pred["point_estimate"]
        ci_lo = pred["ci_lower"]
        ci_hi = pred["ci_upper"]

        # CI shaded band
        fig.add_shape(type="rect", x0=2025.8, x1=2026.2, y0=ci_lo, y1=ci_hi,
                      fillcolor="rgba(124,58,237,0.13)", line_width=0)

        # Prediction point + CI whiskers
        fig.add_trace(go.Scatter(
            x=[2026, 2026, 2026], y=[ci_lo, pe, ci_hi],
            mode="markers+lines",
            name=f"2026 prediction [{ci_lo}–{ci_hi}]",
            line=dict(color="#7c3aed", width=2.5),
            marker=dict(size=[7, 14, 7], color="#7c3aed",
                        symbol=["line-ew", "star", "line-ew"]),
        ))

        fig.add_annotation(
            x=2026, y=ci_hi + max(6, (ci_hi - ci_lo) * 0.15),
            text=f"<b>{pe}</b>", showarrow=False,
            font=dict(size=13, color="#7c3aed"),
        )

    all_pts = [p for p in pts if p is not None]
    y_lo = max(0, min(all_pts, default=0) - 25)
    y_hi = min(650, max(all_pts, default=625) + 35)
    if pred:
        y_lo = min(y_lo, pred["ci_lower"] - 15)
        y_hi = max(y_hi, pred["ci_upper"] + 25)

    fig.update_layout(
        xaxis=dict(title="Year", tickmode="linear",
                   tick0=min(years) if years else 2017, dtick=1,
                   range=[min(years) - 0.5, 2026.5] if years else [2016.5, 2026.5]),
        yaxis=dict(title="CAO Points", range=[max(0, y_lo), min(660, y_hi)]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        margin=dict(l=50, r=20, t=50, b=50),
        height=370,
        hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    return fig


TREND_HTML = {
    "up":     '<span class="trend-up">↑ Rising</span>',
    "down":   '<span class="trend-down">↓ Falling</span>',
    "stable": '<span class="trend-stable">→ Stable</span>',
}

DISCLAIMER = (
    "<b>DISCLAIMER:</b><br>"
    "Predictions will be wrong. This app is vibecoded ai slop that I made for fun. I don't even know how it works — do not base any meaningful decisions off of it."
)


def main():
    st.title("CAO 2026 Vibes-Based Points Predictor")

    st.markdown(f'<div class="disc">{DISCLAIMER}</div>', unsafe_allow_html=True)

    if not ensure_db():
        return

    records = load_data()

    # Build search index
    index = {}   # label -> record
    for r in records:
        label = f"{r['course_code']}  {r['course_name']} — {r['institution']}"
        index[label] = r

    query = st.text_input("Search course name or code", placeholder="e.g. Computer Science, DC116, Medicine …")

    q = query.strip().lower()
    if q:
        matches = [lbl for lbl in index if
                   q in lbl.lower()]
    else:
        matches = []

    if not matches and q:
        st.info("No courses found. Try a different name or code.")
        return

    if not matches:
        st.caption(f"{len(records):,} courses available — type above to search.")
        return

    chosen = st.selectbox("Select course", matches, index=0,
                          label_visibility="collapsed")
    rec  = index[chosen]
    pred = rec["prediction"]

    st.markdown("---")

    # ── Prediction block ──
    if pred:
        trend_html = TREND_HTML.get(pred["trend_direction"], "")
        st.markdown(
            f'<div class="pred-block">'
            f'<div class="pred-num">{pred["point_estimate"]}</div>'
            f'<div class="pred-ci">95% CI &nbsp;[{pred["ci_lower"]} – {pred["ci_upper"]}]</div>'
            f'<div style="margin-top:8px;font-size:1.0em;">{trend_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif rec["is_capped"]:
        st.info("This course requires a Test / Interview / Portfolio — no points-based prediction available.")
    else:
        st.info(f"Insufficient data for a prediction ({rec['n_years_total']} year(s) recorded).")

    # ── Chart ──
    st.plotly_chart(make_chart(rec), use_container_width=True)



if __name__ == "__main__":
    main()
