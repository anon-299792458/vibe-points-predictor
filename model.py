"""
Prediction model for CAO 2026/2027 Point Predictor.

Grading-regime model (updated Nov 2025)
----------------------------------------
Post-marking grade adjustments confirmed by the Dept of Education mean that
2022–2024 were held at approximately 2020 "Calculated Grades" levels — the
same inflation as the COVID years.  The full timeline:

    2017–2019  grade_adjustment_factor = 0.00  (baseline, no uplift)
    2020–2024  grade_adjustment_factor = 1.00  (full inflation plateau)
    2025       grade_adjustment_factor = 0.60  (partial removal, ~midway)
    2026       grade_adjustment_factor = 0.45  (confirmed target, just below 2020)
    2027       grade_adjustment_factor = 0.25  (projected, speculative)

Regression
----------
For each course with ≥3 years of valid data we fit a univariate Theil-Sen
regression on year only, with 2× sample weights for years 2023–2025 (applied
by duplicating those rows before fitting).

For 2026 prediction:  α + β·(2026 − year_mean)
For 2027 projection:  α + β·(2027 − year_mean)  (speculative)

Courses with <3 years of data get history-only display (no prediction).

Confidence intervals
--------------------
80% prediction interval using the t-distribution at n−2 degrees of freedom:

    PI = point_estimate  ±  t(0.90, df=n−2) × σ_residuals × √(1 + 1/n)

With n=3, df=1 → t ≈ 3.08 (wide but honest).
With n≥5, df≥3 → t narrows naturally.

Confidence badge
----------------
    n = 3 or 4  →  "Low confidence (n=X)"
    n ≥ 5       →  "Standard confidence (n=X)"
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import theilslopes, t as t_dist

DB_PATH   = Path(__file__).parent / "predictor.db"

# Grade adjustment factors by year
GRADE_ADJ: dict[int, float] = {
    2017: 0.0, 2018: 0.0, 2019: 0.0,
    2020: 1.0, 2021: 1.0, 2022: 1.0, 2023: 1.0, 2024: 1.0,
    2025: 0.6,
}
GRADE_ADJ_2026 = 0.45   # confirmed government target
GRADE_ADJ_2027 = 0.25   # projected (speculative)

RECENT_YEARS  = {2023, 2024, 2025}
PRED_YEAR     = 2026
MIN_YEARS     = 3
TREND_THRESH  = 3.0     # |slope| pts/year below which trend is 'stable'


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_db():
    conn = sqlite3.connect(DB_PATH)
    df_cp = pd.read_sql_query("SELECT * FROM course_points", conn)
    df_cs = pd.read_sql_query("SELECT * FROM cohort_stats",  conn)
    conn.close()
    return df_cp, df_cs


# ─────────────────────────────────────────────────────────────────────────────
# Cohort median  (for display context only)
# ─────────────────────────────────────────────────────────────────────────────

def _cohort_median(df_year: pd.DataFrame) -> float | None:
    df = df_year.sort_values('points_band_lower').copy()
    total = df['student_count'].sum()
    if total == 0:
        return None
    half = total / 2.0
    cumsum = 0.0
    for _, row in df.iterrows():
        prev_cum = cumsum
        cumsum  += row['student_count']
        if cumsum >= half:
            lower = row['points_band_lower']
            upper = row['points_band_upper']
            band_count = row['student_count']
            if band_count == 0:
                return float((lower + upper) / 2)
            frac = (half - prev_cum) / band_count
            return float(lower + frac * (upper - lower + 1))
    return float(df['points_band_upper'].iloc[-1])


def compute_cohort_medians(df_cs: pd.DataFrame) -> dict[int, float]:
    result = {}
    for year, grp in df_cs.groupby('year'):
        m = _cohort_median(grp)
        if m is not None:
            result[int(year)] = round(m, 1)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-course prediction
# ─────────────────────────────────────────────────────────────────────────────

def _predict(years: np.ndarray, points: np.ndarray,
             grade_adjs: np.ndarray) -> dict | None:
    """
    Fit Theil-Sen (year only) and return 2026 + optional 2027 estimates.
    Requires ≥3 valid data points.  2× weight applied to RECENT_YEARS by
    duplicating those rows before fitting.

    Prediction intervals use the t-distribution at 80% confidence with n−2
    degrees of freedom, giving naturally wide intervals for small n.
    """
    n = len(years)
    if n < MIN_YEARS:
        return None

    # Centre year to improve numerical stability
    year_mean = float(np.mean(years))
    yc = years - year_mean

    # Theil-Sen with 2× weight for recent years via row duplication
    recent_mask = np.isin(years, list(RECENT_YEARS))
    yr_w  = np.concatenate([yc,     yc[recent_mask]])
    pts_w = np.concatenate([points, points[recent_mask]])
    res       = theilslopes(pts_w, yr_w)
    slope     = float(res.slope)
    intercept = float(res.intercept)

    # ── Point estimates ──
    def _pred(pred_year):
        raw = intercept + slope * (pred_year - year_mean)
        return float(np.clip(raw, 0, 625))

    pred_2026 = _pred(PRED_YEAR)
    pred_2027 = _pred(PRED_YEAR + 1)

    # ── Residuals ──
    fitted    = intercept + slope * yc
    residuals = points - fitted
    resid_std = float(np.std(residuals, ddof=1)) if n > 2 else float(np.std(residuals, ddof=0))

    # ── 80% prediction interval via t-distribution at df = n−2 ──
    # t(0.90, df) gives the two-tailed 80% critical value
    df = max(n - 2, 1)
    t_crit = float(t_dist.ppf(0.90, df))
    pi_se_2026 = resid_std * np.sqrt(1 + 1 / n)
    ci_lower   = float(np.clip(pred_2026 - t_crit * pi_se_2026, 0, 625))
    ci_upper   = float(np.clip(pred_2026 + t_crit * pi_se_2026, 0, 625))

    pi_se_2027 = resid_std * np.sqrt(1 + 1 / n)
    ci27_lower = float(np.clip(pred_2027 - t_crit * pi_se_2027, 0, 625))
    ci27_upper = float(np.clip(pred_2027 + t_crit * pi_se_2027, 0, 625))

    # ── Confidence badge ──
    if n <= 4:
        confidence_badge = f"Low confidence (n={n})"
    else:
        confidence_badge = f"Standard confidence (n={n})"

    # ── Trend direction ──
    if   slope >  TREND_THRESH:  trend = 'up'
    elif slope < -TREND_THRESH:  trend = 'down'
    else:                        trend = 'stable'

    # ── Data quality ──
    if   resid_std > 60: quality = 'anomaly-heavy'
    elif n >= 7:         quality = 'good'
    elif n >= 5:         quality = 'limited'
    else:                quality = 'low-data'

    return {
        # 2026
        'point_estimate':   int(round(pred_2026)),
        'ci_lower':         int(round(ci_lower)),
        'ci_upper':         int(round(ci_upper)),
        # 2027 (speculative)
        'pred_2027':        int(round(pred_2027)),
        'ci27_lower':       int(round(ci27_lower)),
        'ci27_upper':       int(round(ci27_upper)),
        # Confidence
        'confidence_badge': confidence_badge,
        # Model diagnostics
        'trend_direction':  trend,
        'data_quality':     quality,
        'slope':            round(slope,     4),
        'intercept':        round(intercept, 2),
        'year_mean':        round(year_mean, 1),
        'residual_std':     round(resid_std,  2),
        'n_years':          n,
        't_crit':           round(t_crit,    3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_predictions() -> tuple[list[dict], dict[int, float], pd.DataFrame]:
    """
    Returns (course_records, cohort_medians, df_cohort_stats).

    Each course_record contains:
        course_code, course_name, institution,
        is_capped, years_data, points_data, is_anomaly_data, grade_adj_data,
        historical {year: points}, n_years_total, prediction | None
    """
    df_cp, df_cs = load_db()

    # Back-fill grade_adjustment_factor if column missing (old DB)
    if 'grade_adjustment_factor' not in df_cp.columns:
        df_cp['grade_adjustment_factor'] = df_cp['year'].map(
            lambda y: GRADE_ADJ.get(int(y), 0.0)
        )
        # Also recompute is_anomaly based on new definition
        df_cp['is_anomaly'] = (df_cp['grade_adjustment_factor'] >= 0.5).astype(int)

    cohort_medians = compute_cohort_medians(df_cs)
    records = []

    for course_code, grp in df_cp.groupby('course_code'):
        grp    = grp.sort_values('year')
        latest = grp.iloc[-1]
        valid  = grp[grp['points'].notna()].copy()

        years_arr     = valid['year'].to_numpy(dtype=float)
        points_arr    = valid['points'].to_numpy(dtype=float)
        anomaly_arr   = valid['is_anomaly'].to_numpy(dtype=float)
        grade_adj_arr = valid['grade_adjustment_factor'].to_numpy(dtype=float)
        is_capped_any = bool(grp['is_capped'].any())

        def _safe_pts(v):
            try:
                return int(v) if v is not None and not (
                    isinstance(v, float) and np.isnan(v)) else None
            except (ValueError, TypeError):
                return None

        historical = {int(r['year']): _safe_pts(r['points']) for _, r in grp.iterrows()}

        rec = {
            'course_code':     str(course_code),
            'course_name':     str(latest['course_name']),
            'institution':     str(latest['institution']) if latest['institution'] else '',
            'is_capped':       is_capped_any,
            'years_data':      years_arr.tolist(),
            'points_data':     points_arr.tolist(),
            'is_anomaly_data': anomaly_arr.tolist(),
            'grade_adj_data':  grade_adj_arr.tolist(),
            'historical':      historical,
            'n_years_total':   len(years_arr),
        }

        if is_capped_any or len(years_arr) == 0:
            rec['prediction'] = None
        else:
            rec['prediction'] = _predict(years_arr, points_arr, grade_adj_arr)

        records.append(rec)

    records.sort(key=lambda r: (r['institution'].lower(), r['course_name'].lower()))
    return records, cohort_medians, df_cs
