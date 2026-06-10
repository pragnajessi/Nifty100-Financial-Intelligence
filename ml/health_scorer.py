"""
ml/health_scorer.py
Computes a 0-100 financial health score for every Nifty 50 company.

Sub-dimension weights:
  Profitability  25%   (net profit margin, OPM%, ROA)
  Revenue Growth 20%   (5Y/3Y CAGR + YoY avg)
  Leverage       20%   (D/E, equity ratio)
  Cash Flow      15%   (FCF consistency, cash conversion, operating cash)
  Dividend       10%   (avg payout, paying years)
  Growth Trend   10%   (revenue + profit slope)

Label thresholds: 85+ EXCELLENT | 70+ GOOD | 50+ AVERAGE | 35+ WEAK | 0+ POOR
"""

import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logger = logging.getLogger(__name__)

WEIGHTS = {
    "profitability": 0.25,
    "growth":        0.20,
    "leverage":      0.20,
    "cashflow":      0.15,
    "dividend":      0.10,
    "trend":         0.10,
}

LABEL_THRESHOLDS = [
    (85, "EXCELLENT"),
    (70, "GOOD"),
    (50, "AVERAGE"),
    (35, "WEAK"),
    (0,  "POOR"),
]

# Nifty 50 banking / NBFC constituents — OPM% is not meaningful for these
BANKING = {
    "AXISBANK", "HDFCBANK", "ICICIBANK", "INDUSINDBK",
    "KOTAKBANK", "SBIN", "BANDHANBNK",
}

# Number of recent fiscal years to use for scoring
SCORING_YEARS = 5


def get_engine():
    """Build a SQLAlchemy engine from environment variables. No hard-coded secrets."""
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "nifty50_warehouse")

    if not db_password:
        raise EnvironmentError(
            "DB_PASSWORD is not set. Please configure it in your .env file."
        )

    url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(url, pool_pre_ping=True)


def load_data(engine):
    """Load the last SCORING_YEARS of financial data for all Nifty 50 companies."""
    year_filter = f"""
        dy.is_ttm = FALSE AND dy.fiscal_year IS NOT NULL
        AND dy.fiscal_year >= (
            SELECT MAX(fiscal_year) - {SCORING_YEARS - 1}
            FROM dim_year WHERE is_ttm = FALSE
        )
    """

    with engine.connect() as conn:
        pl = pd.read_sql(f"""
            SELECT f.symbol, dy.fiscal_year,
                   f.net_profit_margin_pct, f.opm_percentage,
                   f.return_on_assets_pct, f.interest_coverage,
                   f.eps, f.dividend_payout, f.sales, f.net_profit,
                   f.is_banking
            FROM fact_profit_loss f
            JOIN dim_year dy ON f.year_id = dy.year_id
            WHERE {year_filter}
        """, conn)

        bs = pd.read_sql(f"""
            SELECT f.symbol, dy.fiscal_year,
                   f.debt_to_equity, f.equity_ratio, f.borrowings
            FROM fact_balance_sheet f
            JOIN dim_year dy ON f.year_id = dy.year_id
            WHERE {year_filter}
        """, conn)

        cf = pd.read_sql(f"""
            SELECT f.symbol, dy.fiscal_year,
                   f.free_cash_flow, f.cash_conversion_ratio, f.operating_activity
            FROM fact_cash_flow f
            JOIN dim_year dy ON f.year_id = dy.year_id
            WHERE {year_filter}
        """, conn)

        analysis = pd.read_sql(
            "SELECT symbol, period, metric, value_pct FROM fact_analysis",
            conn,
        )

    return pl, bs, cf, analysis


# ── Ranking helpers ───────────────────────────────────────────────────────────

def pct_rank(s: pd.Series) -> pd.Series:
    """Higher raw value → higher score (0–100)."""
    return s.rank(pct=True, na_option="keep") * 100


def pct_rank_inv(s: pd.Series) -> pd.Series:
    """Lower raw value → higher score (0–100). Used for D/E ratio."""
    return (1 - s.rank(pct=True, na_option="keep")) * 100


# ── Sub-dimension scorers ─────────────────────────────────────────────────────

def score_profitability(pl: pd.DataFrame) -> pd.Series:
    """
    Profitability score (25% weight).

    Banking companies have NULL OPM% (not applicable). Their OPM rank is
    replaced with a neutral 50th-percentile value so they are not penalised.
    """
    agg = pl.groupby("symbol").agg(
        avg_npm=("net_profit_margin_pct", "mean"),
        avg_opm=("opm_percentage",        "mean"),
        avg_roa=("return_on_assets_pct",  "mean"),
    )
    opm_rank = pct_rank(agg["avg_opm"]).fillna(50)
    s = (
        pct_rank(agg["avg_npm"]) * 0.45
        + opm_rank               * 0.25
        + pct_rank(agg["avg_roa"]) * 0.30
    )
    return s.rename("profitability_score")


def score_growth(pl: pd.DataFrame, analysis: pd.DataFrame) -> pd.Series:
    """
    Revenue-growth score (20% weight).

    Uses compounded_sales_growth from fact_analysis for 5Y and 3Y CAGR,
    plus a simple YoY average computed from the raw P&L rows.
    Missing CAGR periods fall back to the 50th-percentile neutral value.
    """
    an = analysis[analysis["metric"] == "compounded_sales_growth"]
    pivot = an.pivot_table(index="symbol", columns="period", values="value_pct")

    pl_sorted = pl.sort_values(["symbol", "fiscal_year"])
    pl_sorted["yoy"] = pl_sorted.groupby("symbol")["sales"].pct_change() * 100
    avg_yoy = pl_sorted.groupby("symbol")["yoy"].mean()

    score = pd.Series(0.0, index=avg_yoy.index)
    if "5Y" in pivot.columns:
        score = score.add(pct_rank(pivot["5Y"]).fillna(50) * 0.45, fill_value=0)
    if "3Y" in pivot.columns:
        score = score.add(pct_rank(pivot["3Y"]).fillna(50) * 0.25, fill_value=0)
    score = score.add(pct_rank(avg_yoy).fillna(50) * 0.30, fill_value=0)

    return score.rename("growth_score")


def score_leverage(bs: pd.DataFrame) -> pd.Series:
    """
    Leverage score (20% weight).

    Debt-free companies (D/E = 0) naturally rank in the top percentile.
    """
    agg = bs.groupby("symbol").agg(
        avg_de=("debt_to_equity", "mean"),
        avg_er=("equity_ratio",   "mean"),
    )
    s = (
        pct_rank_inv(agg["avg_de"]).fillna(50) * 0.60
        + pct_rank(agg["avg_er"]).fillna(50)   * 0.40
    )
    return s.rename("leverage_score")


def score_cashflow(cf: pd.DataFrame) -> pd.Series:
    """
    Cash-flow score (15% weight).

    Rewards companies that generate free cash flow in most years, maintain a
    high cash-conversion ratio, and produce strong operating cash flow.
    """
    agg = cf.groupby("symbol").agg(
        pos_fcf=("free_cash_flow",        lambda x: (x > 0).sum()),
        avg_ccr=("cash_conversion_ratio", "mean"),
        avg_op =("operating_activity",    "mean"),
    )
    s = (
        pct_rank(agg["pos_fcf"]).fillna(50) * 0.40
        + pct_rank(agg["avg_ccr"]).fillna(50) * 0.35
        + pct_rank(agg["avg_op"]).fillna(50)  * 0.25
    )
    return s.rename("cashflow_score")


def score_dividend(pl: pd.DataFrame) -> pd.Series:
    """
    Dividend score (10% weight).

    Rewards consistent dividend payers; non-payers score low but are not
    penalised beyond the bottom percentile.
    """
    agg = pl.groupby("symbol").agg(
        avg_payout=("dividend_payout", "mean"),
        paying_yrs=("dividend_payout", lambda x: (x > 0).sum()),
    )
    s = (
        pct_rank(agg["avg_payout"]).fillna(50) * 0.50
        + pct_rank(agg["paying_yrs"]).fillna(50) * 0.50
    )
    return s.rename("dividend_score")


def score_trend(pl: pd.DataFrame) -> pd.Series:
    """
    Growth-trend score (10% weight).

    Fits a linear slope over recent fiscal years for both revenue and net
    profit margin. Requires at least 3 data points; otherwise returns NaN
    (filled with 50 in the final combination step).
    """
    def _slope(series: pd.Series) -> float:
        series = series.dropna()
        if len(series) < 3:
            return np.nan
        return np.polyfit(np.arange(len(series)), series.values, 1)[0]

    pl_sorted = pl.sort_values(["symbol", "fiscal_year"])
    sales_slope  = pl_sorted.groupby("symbol")["sales"].apply(_slope)
    margin_slope = pl_sorted.groupby("symbol")["net_profit_margin_pct"].apply(_slope)
    s = (
        pct_rank(sales_slope).fillna(50)  * 0.50
        + pct_rank(margin_slope).fillna(50) * 0.50
    )
    return s.rename("trend_score")


# ── Label assignment ──────────────────────────────────────────────────────────

def assign_label(score: float) -> str:
    for threshold, label in LABEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "POOR"


# ── Main computation ──────────────────────────────────────────────────────────

def compute_scores(engine=None) -> pd.DataFrame:
    """
    Compute health scores for every company in the warehouse.

    Returns a DataFrame with columns:
        symbol, overall_score, profitability_score, growth_score,
        leverage_score, cashflow_score, dividend_score, trend_score,
        health_label, computed_at
    """
    if engine is None:
        engine = get_engine()

    pl, bs, cf, analysis = load_data(engine)

    if pl.empty:
        logger.warning("No P&L data found — check your database connection and ETL pipeline.")
        return pd.DataFrame()

    sub_scores = [
        score_profitability(pl),
        score_growth(pl, analysis),
        score_leverage(bs),
        score_cashflow(cf),
        score_dividend(pl),
        score_trend(pl),
    ]
    combined = pd.concat(sub_scores, axis=1).fillna(50)

    combined["overall_score"] = (
        combined["profitability_score"] * WEIGHTS["profitability"]
        + combined["growth_score"]      * WEIGHTS["growth"]
        + combined["leverage_score"]    * WEIGHTS["leverage"]
        + combined["cashflow_score"]    * WEIGHTS["cashflow"]
        + combined["dividend_score"]    * WEIGHTS["dividend"]
        + combined["trend_score"]       * WEIGHTS["trend"]
    ).clip(0, 100).round(2)

    combined["health_label"] = combined["overall_score"].apply(assign_label)
    combined["computed_at"]  = datetime.now(timezone.utc).replace(tzinfo=None)

    return combined.reset_index().rename(columns={"index": "symbol"})


def save_scores(scores: pd.DataFrame, engine=None) -> None:
    """
    Upsert ML scores into fact_ml_scores.

    Uses ON CONFLICT to update all score columns if the (symbol, computed_at)
    pair already exists (e.g. re-running on the same UTC minute).
    """
    if engine is None:
        engine = get_engine()

    if scores.empty:
        logger.warning("No scores to save.")
        return

    cols = [
        "symbol", "computed_at", "overall_score", "profitability_score",
        "growth_score", "leverage_score", "cashflow_score",
        "dividend_score", "trend_score", "health_label",
    ]
    scores = scores[cols]

    with engine.begin() as conn:
        for _, row in scores.iterrows():
            conn.execute(text("""
                INSERT INTO fact_ml_scores
                    (symbol, computed_at, overall_score, profitability_score,
                     growth_score, leverage_score, cashflow_score,
                     dividend_score, trend_score, health_label)
                VALUES
                    (:symbol, :computed_at, :overall_score, :profitability_score,
                     :growth_score, :leverage_score, :cashflow_score,
                     :dividend_score, :trend_score, :health_label)
                ON CONFLICT (symbol, computed_at) DO UPDATE SET
                    overall_score       = EXCLUDED.overall_score,
                    profitability_score = EXCLUDED.profitability_score,
                    growth_score        = EXCLUDED.growth_score,
                    leverage_score      = EXCLUDED.leverage_score,
                    cashflow_score      = EXCLUDED.cashflow_score,
                    dividend_score      = EXCLUDED.dividend_score,
                    trend_score         = EXCLUDED.trend_score,
                    health_label        = EXCLUDED.health_label
            """), row.to_dict())

    logger.info("Saved %d health scores to fact_ml_scores.", len(scores))
    print(f"✓ Saved {len(scores)} health scores to fact_ml_scores.")


if __name__ == "__main__":
    engine = get_engine()
    scores = compute_scores(engine)

    if scores.empty:
        print("No scores computed. Exiting.")
    else:
        print("\nTop 10 Nifty 50 companies by health score:")
        print(
            scores[["symbol", "overall_score", "health_label"]]
            .sort_values("overall_score", ascending=False)
            .head(10)
            .to_string(index=False)
        )
        print("\nLabel distribution:")
        print(scores["health_label"].value_counts().to_string())
        save_scores(scores, engine)
