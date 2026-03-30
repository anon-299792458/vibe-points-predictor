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
For each course with ≥3 years of valid data we fit OLS (ordinary least
squares) on year only.  Recent years (2023–2025) receive 2× weight, applied
by duplicating those rows before fitting.  OLS is used (not Theil-Sen) so that
the t-distribution prediction interval formula is statistically valid.

For 2026 prediction:  α + β·year
For 2027 projection:  α + β·year  (speculative)

Courses with <3 years of data get history-only display (no prediction).

Confidence intervals
--------------------
Full 80% OLS prediction interval, including the extrapolation leverage term:

    s   = √( Σeᵢ² / (n−2) )          residual std on original n points
    h   = 1/n + (x_new − x̄)² / Sxx   leverage at the prediction point
    PI  = ŷ  ±  t(0.90, df=n−2) × s × √(1 + h)

The leverage term (x_new − x̄)²/Sxx widens the interval honestly for
extrapolation — predicting 2026 from data ending in 2025 always incurs this.

With n=3, df=1 → t ≈ 3.08, and leverage for a 1-year extrapolation ≈ 4+,
giving a very wide but honest interval.  With n≥8 the interval narrows
substantially.

Confidence badge
----------------
    n = 3 or 4  →  "Low confidence (n=X)"
    n ≥ 5       →  "Standard confidence (n=X)"
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

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
    Fit OLS (year only) and return 2026 + 2027 estimates with 80% prediction
    intervals.  Requires ≥3 valid data points.

    Recent years (RECENT_YEARS) receive 2× weight by row duplication before
    fitting.  Residuals and PI are computed on the original n observations so
    that degrees of freedom reflect actual data, not the duplicated set.

    PI formula (full OLS, includes extrapolation leverage):
        s     = sqrt( sum(eᵢ²) / (n-2) )
        h_new = 1/n + (x_new - x̄)² / Sxx
        PI    = ŷ ± t(0.90, n-2) × s × sqrt(1 + h_new)
    """
    n = len(years)
    if n < MIN_YEARS:
        return None

    year_mean = float(np.mean(years))

    # ── OLS fit on row-duplicated data (2× weight for recent years) ──
    recent_mask = np.isin(years, list(RECENT_YEARS))
    yr_w  = np.concatenate([years, years[recent_mask]])
    pts_w = np.concatenate([points, points[recent_mask]])
    slope, intercept = np.polyfit(yr_w, pts_w, 1)
    slope     = float(slope)
    intercept = float(intercept)

    # ── Point estimates ──
    def _pred(x):
        return float(np.clip(intercept + slope * x, 0, 625))

    pred_2026 = _pred(PRED_YEAR)
    pred_2027 = _pred(PRED_YEAR + 1)

    # ── Residuals on the original n points, df = n-2 ──
    fitted    = intercept + slope * years
    residuals = points - fitted
    df        = max(n - 2, 1)
    s         = float(np.sqrt(np.sum(residuals ** 2) / df))

    # ── Full OLS prediction interval with extrapolation leverage ──
    # h_new = 1/n + (x_new - x_bar)^2 / Sxx
    Sxx = float(np.sum((years - year_mean) ** 2))

    def _pi_half(x_new):
        h = (1.0 / n) + ((x_new - year_mean) ** 2 / Sxx if Sxx > 0 else 0.0)
        return s * float(np.sqrt(1.0 + h))

    # t(0.90, df) is the 80% two-tailed critical value
    t_crit = float(t_dist.ppf(0.90, df))

    ci_lower   = float(np.clip(pred_2026 - t_crit * _pi_half(PRED_YEAR),     0, 625))
    ci_upper   = float(np.clip(pred_2026 + t_crit * _pi_half(PRED_YEAR),     0, 625))
    ci27_lower = float(np.clip(pred_2027 - t_crit * _pi_half(PRED_YEAR + 1), 0, 625))
    ci27_upper = float(np.clip(pred_2027 + t_crit * _pi_half(PRED_YEAR + 1), 0, 625))

    # ── Confidence badge ──
    confidence_badge = (
        f"Low confidence (n={n})" if n <= 4 else f"Standard confidence (n={n})"
    )

    # ── Trend direction ──
    if   slope >  TREND_THRESH:  trend = 'up'
    elif slope < -TREND_THRESH:  trend = 'down'
    else:                        trend = 'stable'

    # ── Data quality ──
    if   s    > 60: quality = 'anomaly-heavy'
    elif n   >= 7:  quality = 'good'
    elif n   >= 5:  quality = 'limited'
    else:           quality = 'low-data'

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
        'residual_std':     round(s,          2),
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
