"""
AlphaLens AI – Scoring Engine
==============================
Produces a composite stock score (0–100) from six weighted dimensions,
plus confidence, risk, forecast, and an AI-generated natural-language summary.

Dimensions & weights:
  1. Growth Quality       28 %
  2. Profitability         22 %
  3. Balance Sheet         15 %
  4. Free Cash Flow Quality 20 %
  5. Momentum               8 %
  6. Market Signals          7 %

The module is importable by the pipeline **and** runnable standalone for
quick smoke-testing:  python engine/scorer.py
"""

from __future__ import annotations

import math
import logging
from typing import Any

import numpy as np

logger = logging.getLogger("alphalens.scorer")


# ═══════════════════════════════════════════════════════════════════════════
#  Helper utilities
# ═══════════════════════════════════════════════════════════════════════════

def _vals(series: list[dict[str, Any]]) -> list[float]:
    """Extract numeric values from a metric time-series."""
    return [float(pt["val"]) for pt in series if pt.get("val") is not None]


def _last_n(series: list[dict[str, Any]], n: int = 8) -> list[float]:
    """Return the last *n* numeric values."""
    return _vals(series)[-n:]


def _cagr(values: list[float]) -> float:
    """Compound annual growth rate over *values* (assumed quarterly)."""
    if len(values) < 2 or values[0] == 0:
        return 0.0
    periods = len(values) - 1
    end, start = values[-1], values[0]
    if start <= 0 or end <= 0:
        return 0.0
    return (end / start) ** (4.0 / periods) - 1.0  # annualized


def _variance_penalty(values: list[float]) -> float:
    """Return 0–1 penalty based on coefficient of variation."""
    if len(values) < 3:
        return 0.0
    arr = np.array(values, dtype=float)
    mean = np.mean(arr)
    if mean == 0:
        return 0.5
    cv = float(np.std(arr) / abs(mean))
    return min(cv, 1.0)


def _positive_ratio(values: list[float]) -> float:
    """Fraction of values that are > 0."""
    if not values:
        return 0.0
    return sum(1 for v in values if v > 0) / len(values)


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0:
        return default
    return a / b


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _sigmoid_score(x: float, midpoint: float = 0.0, steepness: float = 10.0) -> float:
    """Map a real number to 0–100 via a sigmoid centred on *midpoint*."""
    try:
        return 100.0 / (1.0 + math.exp(-steepness * (x - midpoint)))
    except OverflowError:
        return 0.0 if x < midpoint else 100.0


# ═══════════════════════════════════════════════════════════════════════════
#  Dimension scorers  (each returns 0–100)
# ═══════════════════════════════════════════════════════════════════════════

def _score_growth(metrics: dict) -> float:
    """Growth Quality dimension (28 %)."""
    rev = _last_n(metrics.get("revenue", []))
    ni = _last_n(metrics.get("net_income", []))

    rev_cagr = _cagr(rev)
    rev_stability = 1.0 - _variance_penalty(rev)
    ni_growth = _cagr(ni)

    # Weighted sub-score: CAGR is most important, stability adds bonus
    raw = (
        _sigmoid_score(rev_cagr, midpoint=0.08, steepness=15) * 0.50
        + rev_stability * 100 * 0.25
        + _sigmoid_score(ni_growth, midpoint=0.05, steepness=12) * 0.25
    )
    return _clamp(raw)


def _score_profitability(metrics: dict) -> float:
    """Profitability dimension (22 %)."""
    rev = _last_n(metrics.get("revenue", []))
    gp = _last_n(metrics.get("gross_profit", []))
    oi = _last_n(metrics.get("operating_income", []))
    ta = _last_n(metrics.get("total_assets", []))

    gross_margin = _safe_div(gp[-1], rev[-1]) if gp and rev else 0.0
    op_margin = _safe_div(oi[-1], rev[-1]) if oi and rev else 0.0
    roic = _safe_div(oi[-1], ta[-1]) if oi and ta else 0.0

    raw = (
        _sigmoid_score(gross_margin, 0.35, 8) * 0.35
        + _sigmoid_score(op_margin, 0.15, 10) * 0.35
        + _sigmoid_score(roic, 0.08, 12) * 0.30
    )
    return _clamp(raw)


def _score_balance_sheet(metrics: dict) -> float:
    """Balance Sheet dimension (15 %)."""
    rev = _last_n(metrics.get("revenue", []))
    debt = _last_n(metrics.get("total_debt", []))
    cash = _last_n(metrics.get("cash", []))
    ta = _last_n(metrics.get("total_assets", []))
    ie = _last_n(metrics.get("interest_expense", []))
    oi = _last_n(metrics.get("operating_income", []))

    net_debt_rev = 0.0
    if debt and cash and rev and rev[-1] != 0:
        net_debt_rev = (debt[-1] - cash[-1]) / rev[-1]

    interest_coverage = 0.0
    if oi and ie and ie[-1] > 0:
        interest_coverage = oi[-1] / ie[-1]

    cash_assets = 0.0
    if cash and ta and ta[-1] > 0:
        cash_assets = cash[-1] / ta[-1]

    raw = (
        _sigmoid_score(-net_debt_rev, -0.5, 5) * 0.40    # lower debt is better
        + _sigmoid_score(interest_coverage, 5.0, 1.0) * 0.35
        + _sigmoid_score(cash_assets, 0.10, 15) * 0.25
    )
    return _clamp(raw)


def _score_fcf(metrics: dict) -> float:
    """Free Cash Flow Quality dimension (20 %)."""
    ocf = _last_n(metrics.get("operating_cash_flow", []))
    capex = _last_n(metrics.get("capex", []))
    ni = _last_n(metrics.get("net_income", []))

    min_len = min(len(ocf), len(capex))
    if min_len == 0:
        return 50.0  # neutral when no data

    fcf = [ocf[i] - capex[i] for i in range(min_len)]

    fcf_cagr = _cagr([{"val": v} for v in fcf] if False else [])
    # Build proper series for _cagr
    fcf_series = [{"val": v} for v in fcf]
    fcf_cagr = _cagr(_vals(fcf_series)) if len(fcf) >= 2 else 0.0
    # Recompute using raw list
    if len(fcf) >= 2 and fcf[0] > 0 and fcf[-1] > 0:
        periods = len(fcf) - 1
        fcf_cagr = (fcf[-1] / fcf[0]) ** (4.0 / periods) - 1.0
    else:
        fcf_cagr = 0.0

    fcf_ni_conv = 0.0
    if ni and ni[-1] != 0 and fcf:
        fcf_ni_conv = fcf[-1] / abs(ni[-1])

    fcf_consistency = 1.0 - _variance_penalty(fcf) if len(fcf) >= 3 else 0.5
    pos_ratio = _positive_ratio(fcf)

    raw = (
        _sigmoid_score(fcf_cagr, 0.06, 12) * 0.30
        + _sigmoid_score(fcf_ni_conv, 0.8, 5) * 0.25
        + fcf_consistency * 100 * 0.20
        + pos_ratio * 100 * 0.25
    )
    return _clamp(raw)


def _score_momentum(market: dict) -> float:
    """Momentum dimension (8 %)."""
    pvs50 = market.get("price_vs_sma50", 0.0) or 0.0
    pvs200 = market.get("price_vs_sma200", 0.0) or 0.0
    rsi = market.get("rsi_14", 50.0) or 50.0

    # Normalize RSI: 50 → 50, 30 → 20, 70 → 80
    rsi_norm = _clamp((rsi - 30) / 40 * 100)

    raw = (
        _sigmoid_score(pvs50, 0.0, 20) * 0.35
        + _sigmoid_score(pvs200, 0.0, 12) * 0.35
        + rsi_norm * 0.30
    )
    return _clamp(raw)


def _score_market_signals(market: dict) -> float:
    """Market Signals dimension (7 %)."""
    opt = market.get("options_sentiment", "neutral")
    insider = market.get("insider_activity", "unknown")

    opt_score = {"bullish": 80, "neutral": 50, "bearish": 20}.get(opt, 50)
    ins_score = {"net_buying": 85, "neutral": 50, "net_selling": 20, "unknown": 50}.get(insider, 50)

    raw = opt_score * 0.50 + ins_score * 0.50
    return _clamp(raw)


# ═══════════════════════════════════════════════════════════════════════════
#  Composite scoring
# ═══════════════════════════════════════════════════════════════════════════

WEIGHTS = {
    "growth":         0.28,
    "profitability":  0.22,
    "balance_sheet":  0.15,
    "fcf":            0.20,
    "momentum":       0.08,
    "market_signals": 0.07,
}

RATING_THRESHOLDS = [
    (78, "STRONG BUY"),
    (63, "BUY"),
    (48, "NEUTRAL"),
    (33, "CAUTION"),
    (0,  "RISK"),
]


def _rating_label(score: float) -> str:
    for threshold, label in RATING_THRESHOLDS:
        if score >= threshold:
            return label
    return "RISK"


# ═══════════════════════════════════════════════════════════════════════════
#  Confidence score
# ═══════════════════════════════════════════════════════════════════════════

def _compute_confidence(metrics: dict, market: dict) -> float:
    """Confidence = percentage of expected data that is actually available."""
    total_checks = 0
    passed = 0

    # Check each metric has at least 4 quarters
    for key in [
        "revenue", "gross_profit", "operating_income", "net_income",
        "operating_cash_flow", "capex", "total_debt", "cash",
        "total_assets", "interest_expense", "eps_diluted",
    ]:
        total_checks += 1
        vals = _last_n(metrics.get(key, []))
        if len(vals) >= 4:
            passed += 1
        elif len(vals) >= 1:
            passed += 0.5

    # Bonus for 8+ quarters of revenue
    total_checks += 1
    if len(_last_n(metrics.get("revenue", []), 8)) >= 8:
        passed += 1

    # Polygon availability
    total_checks += 1
    if not market.get("stub", True):
        passed += 1
    else:
        passed += 0.3  # partial credit for stubs

    return _clamp(passed / total_checks * 100)


# ═══════════════════════════════════════════════════════════════════════════
#  Forecast model
# ═══════════════════════════════════════════════════════════════════════════

def _compute_forecast(composite: float, metrics: dict, market: dict | None = None) -> dict[str, float]:
    rev = _last_n(metrics.get("revenue", []))
    rev_cagr = _cagr(rev)

    # Score component: composite 50 = 0%, composite 100 = +20%
    score_component = (composite - 50) * 0.004

    # Revenue component: diminishing returns via log so hypergrowth
    # stocks (NVDA at 150% CAGR) don't generate absurd forecasts.
    # Positive growth: log-scaled, capped at 0.5 before log.
    # Negative growth: small linear penalty.
    if rev_cagr > 0:
        rev_component = math.log(1.0 + min(rev_cagr, 0.50)) * 0.30
    else:
        rev_component = rev_cagr * 0.15

    base_return = score_component + rev_component

    # Valuation discount: high price-to-sales stocks are already pricing
    # in a lot of the good news, so we haircut the forecast.
    if market:
        market_cap = market.get("market_cap") or 0
        annual_rev = rev[-1] * 4 if rev else 0
        if market_cap > 0 and annual_rev > 0:
            ps_ratio = market_cap / annual_rev
            if ps_ratio > 15:
                # For every 10x of P/S above 15, shave ~3% off the base return
                valuation_discount = min((ps_ratio - 15) / 10.0 * 0.03, 0.12)
                base_return -= valuation_discount

    # Hard cap: realistic 12-month range
    base_return = max(-0.30, min(0.40, base_return))

    # Bull adds ~20 percentage points; bear subtracts ~25 percentage points
    bull_return = min(base_return + 0.20, 0.55)
    bear_return = max(base_return - 0.25, -0.40)

    return {
        "base_return_pct": round(base_return * 100, 1),
        "bull_return_pct": round(bull_return * 100, 1),
        "bear_return_pct": round(bear_return * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Risk score
# ═══════════════════════════════════════════════════════════════════════════

def _compute_risk(composite: float, metrics: dict) -> float:
    rev = _last_n(metrics.get("revenue", []))
    debt = _last_n(metrics.get("total_debt", []))
    rev_vol_penalty = _variance_penalty(rev) * 15

    leverage_penalty = 0.0
    if debt and rev and rev[-1] > 0:
        leverage_ratio = debt[-1] / rev[-1]
        leverage_penalty = min(leverage_ratio * 10, 20)

    risk = 100 - composite * 0.6 + rev_vol_penalty + leverage_penalty
    return _clamp(risk)


# ═══════════════════════════════════════════════════════════════════════════
#  AI Summary (deterministic rule-based templates)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_summary(
    ticker: str,
    dimensions: dict[str, float],
    metrics: dict,
    market: dict,
    composite: float,
    rating: str,
) -> str:
    parts: list[str] = []

    # Revenue growth
    rev = _last_n(metrics.get("revenue", []))
    rev_cagr = _cagr(rev)
    if rev_cagr > 0.15:
        parts.append(f"{ticker} is delivering strong revenue growth at ~{rev_cagr:.0%} annualized.")
    elif rev_cagr > 0.05:
        parts.append(f"{ticker} shows moderate revenue growth (~{rev_cagr:.0%} CAGR).")
    elif rev_cagr > 0:
        parts.append(f"{ticker} has slow but positive revenue growth ({rev_cagr:.0%} CAGR).")
    else:
        parts.append(f"{ticker} revenue is declining ({rev_cagr:.0%} CAGR).")

    # Margin trend
    prof_score = dimensions.get("profitability", 50)
    if prof_score >= 70:
        parts.append("Margins are healthy with strong operating efficiency.")
    elif prof_score >= 45:
        parts.append("Margins are at an acceptable level.")
    else:
        parts.append("Margins are under pressure.")

    # FCF quality
    fcf_score = dimensions.get("fcf", 50)
    if fcf_score >= 70:
        parts.append("Free cash flow generation is robust and consistent.")
    elif fcf_score >= 45:
        parts.append("Free cash flow is adequate but could improve.")
    else:
        parts.append("Free cash flow quality is weak, warranting caution.")

    # Insider activity
    insider = market.get("insider_activity", "unknown")
    if insider == "net_buying":
        parts.append("Insider buying signals management confidence.")
    elif insider == "net_selling":
        parts.append("Notable insider selling has been observed.")
    else:
        parts.append("Insider activity is unremarkable.")

    # Options sentiment
    opt = market.get("options_sentiment", "neutral")
    if opt == "bullish":
        parts.append("Options flow tilts bullish.")
    elif opt == "bearish":
        parts.append("Options market shows bearish positioning.")

    # Final verdict
    parts.append(f"Overall rating: {rating} ({composite:.0f}/100).")

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
#  Company profiles (curated descriptions for major tickers)
# ═══════════════════════════════════════════════════════════════════════════

COMPANY_PROFILES: dict[str, str] = {
    "AAPL":  "Apple designs and sells consumer electronics, software, and services. Its hardware lineup — iPhone, Mac, iPad, AirPods, Apple Watch — is complemented by a fast-growing Services segment (App Store, iCloud, Apple Pay) that now represents roughly 25% of revenue and carries significantly higher margins than hardware.",
    "MSFT":  "Microsoft provides cloud infrastructure (Azure), productivity software (Office 365), and developer tools (GitHub, VS Code). Azure is the primary growth engine, competing directly with AWS and Google Cloud in a market expected to exceed $1 trillion, while the Copilot AI suite is being embedded across every product line.",
    "NVDA":  "NVIDIA designs GPUs originally built for gaming that have become the dominant hardware for AI training and inference. Its CUDA software ecosystem creates deep switching costs, and its data center segment now dwarfs gaming revenue as hyperscalers race to build AI infrastructure at scale.",
    "GOOGL": "Alphabet operates Google Search, YouTube, Google Cloud, and a suite of advertising products that collectively reach billions of users daily. Search advertising remains the dominant profit driver, while Google Cloud is scaling rapidly toward profitability and DeepMind's Gemini models represent a significant AI asset.",
    "GOOG":  "Alphabet operates Google Search, YouTube, Google Cloud, and a suite of advertising products that collectively reach billions of users daily. Search advertising remains the dominant profit driver, while Google Cloud is scaling rapidly toward profitability and DeepMind's Gemini models represent a significant AI asset.",
    "AMZN":  "Amazon operates the world's largest e-commerce marketplace alongside AWS, which generates the majority of operating profit despite being a fraction of total revenue. The combination of logistics network scale and cloud dominance creates a dual-engine business model with compounding advantages and multiple expansion levers.",
    "META":  "Meta operates Facebook, Instagram, and WhatsApp — the largest social media ecosystem on earth — monetized almost entirely through targeted digital advertising. It is also investing heavily in AI infrastructure and Reality Labs, which pressures near-term margins but positions it for emerging compute and social platforms.",
    "TSLA":  "Tesla designs and manufactures battery electric vehicles and energy storage systems, and holds the highest gross margins in the traditional auto industry. It has a growing software and services segment (Full Self-Driving, Supercharging), but faces intensifying competition from Chinese EV makers and legacy OEMs.",
    "JPM":   "JPMorgan Chase is the largest U.S. bank by assets, operating consumer banking, investment banking, commercial banking, and asset management divisions. It is widely regarded as best-in-class for risk management, with a fortress balance sheet and the highest return on equity among mega-cap banks.",
    "V":     "Visa operates the world's largest retail electronic payments network, earning transaction and service fees without taking on credit risk. Its business is an asset-light toll booth on global commerce — operating margins exceed 60% and network effects make displacement nearly impossible.",
    "MA":    "Mastercard runs the second-largest global payments network and earns fees on every transaction processed across its rails. Like Visa, it is asset-light and highly profitable, with a structurally advantaged position in the secular shift from cash to digital payments.",
    "BRK.B": "Berkshire Hathaway is a conglomerate controlled by Warren Buffett that owns wholly or partially businesses across insurance (GEICO, Gen Re), railroads (BNSF), utilities (BHE), manufacturing, and a large publicly traded equity portfolio dominated by Apple and financial stocks.",
    "UNH":   "UnitedHealth Group is the largest U.S. health insurer and operates Optum, a fast-growing health services division covering pharmacy benefits, care delivery, and health IT. It is a vertically integrated health system with a track record of consistent mid-teens EPS growth across economic cycles.",
    "LLY":   "Eli Lilly is a pharmaceutical company best known for its GLP-1 drug franchise (Mounjaro, Zepbound) for diabetes and obesity, which is reshaping the company's revenue trajectory. It is one of the fastest-growing large-cap pharma companies and is investing aggressively in manufacturing capacity to meet surging demand.",
    "AVGO":  "Broadcom designs semiconductors and infrastructure software for data centers, networking, and wireless connectivity. Its acquisition of VMware layers high-margin recurring software revenue on top of its chip business, and its custom AI chip relationships with hyperscalers are a growing revenue driver.",
    "ORCL":  "Oracle provides database software, cloud infrastructure (OCI), and enterprise applications (ERP/CRM via NetSuite and Fusion Cloud). It is in the middle of a multi-year cloud transition that is accelerating revenue growth, with AI-driven database and infrastructure demand becoming a meaningful tailwind.",
    "NFLX":  "Netflix is the world's leading subscription video streaming platform with over 300 million paid memberships globally. It re-accelerated growth through password-sharing enforcement and an ad-supported tier, and is now investing in live events and gaming to reduce content-driven churn.",
    "AMD":   "Advanced Micro Devices designs CPUs and GPUs for data centers, gaming, and embedded systems. It has taken significant server CPU share from Intel over the past five years and is now competing with NVIDIA in AI accelerator chips with its MI300X series.",
    "INTC":  "Intel designs and manufactures CPUs for PCs and data center servers. It is undergoing a costly strategic transformation — rebuilding its process technology leadership and launching a contract foundry (Intel Foundry Services) — while losing share to AMD in CPUs and NVIDIA in AI.",
    "CRM":   "Salesforce is the leading cloud CRM platform, providing sales, service, marketing, and analytics software to enterprise customers on a subscription basis. It is integrating AI agents (Agentforce) across its platform to protect its market position as software procurement patterns shift toward outcomes-based buying.",
    "TSMC":  "TSMC is the world's dominant contract semiconductor manufacturer, producing chips designed by Apple, NVIDIA, AMD, Qualcomm, and others. As the only company capable of manufacturing the most advanced chips at scale, it sits at the center of the global AI infrastructure buildout.",
    "SHOP":  "Shopify provides e-commerce software and payment infrastructure for small businesses to enterprise merchants globally. It has expanded from a storefront platform into a full commerce operating system — spanning logistics, point-of-sale, lending, and cross-border payments — making merchant switching costs very high.",
    "SPOT":  "Spotify is the world's largest audio streaming platform with over 600 million monthly active users across music, podcasts, and audiobooks. After years of thin margins, it is now converting scale into profit through price increases, podcast monetization, and its expanding creator ecosystem.",
    "COIN":  "Coinbase is the largest regulated cryptocurrency exchange in the United States, earning transaction fees and subscription revenue from retail and institutional crypto traders. Its revenue is highly correlated with crypto market volatility and Bitcoin price cycles, making it a high-beta proxy on digital asset sentiment.",
    "PLTR":  "Palantir builds AI and data analytics platforms used by government defense agencies (Gotham) and commercial enterprises (Foundry, AIP). Its AI Platform is gaining commercial momentum and it holds a structurally advantaged position in U.S. government AI contracts as defense budgets shift toward software.",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Company Overview (narrative context for the detail panel)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_overview(
    ticker: str,
    dimensions: dict[str, float],
    metrics: dict,
    market: dict,
    composite: float,
    rating: str,
) -> dict[str, str]:
    """Return a dict with description, tailwinds, risks, and verdict strings."""

    # --- Company description ---
    description = COMPANY_PROFILES.get(ticker.upper())
    if not description:
        rev = _last_n(metrics.get("revenue", []))
        gp = _last_n(metrics.get("gross_profit", []))
        gross_margin = _safe_div(gp[-1], rev[-1]) if gp and rev else 0.0
        if gross_margin > 0.60:
            biz_type = "a high-margin software or services business"
        elif gross_margin > 0.40:
            biz_type = "a product or mixed-revenue business with solid margins"
        elif gross_margin > 0.20:
            biz_type = "a product-heavy business with moderate margins"
        else:
            biz_type = "a capital-intensive or low-margin business"
        rev_q = rev[-1] if rev else 0.0
        if rev_q >= 1e9:
            rev_display = f"${rev_q / 1e9:.1f}B"
        elif rev_q >= 1e6:
            rev_display = f"${rev_q / 1e6:.0f}M"
        else:
            rev_display = "N/A"
        description = (
            f"{ticker} appears to be {biz_type}, reporting {rev_display} in its most recent quarter. "
            f"Financial data is sourced from SEC 10-Q/10-K filings."
        )

    # --- Tailwinds (top 2 scoring dimensions) ---
    dim_positive = {
        "growth":         "accelerating revenue growth",
        "profitability":  "strong and expanding margins",
        "balance_sheet":  "a clean balance sheet with low debt",
        "fcf":            "high-quality free cash flow conversion",
        "momentum":       "bullish price momentum",
        "market_signals": "positive institutional and options sentiment",
    }
    sorted_dims = sorted(dimensions.items(), key=lambda x: x[1], reverse=True)
    top_parts = [dim_positive[k] for k, v in sorted_dims[:2] if v >= 55]
    if not top_parts:
        top_parts = [dim_positive[sorted_dims[0][0]]]
    tailwinds = "The primary performance driver is " + " and ".join(top_parts) + "."

    # --- Risk (weakest dimension, with a leverage override) ---
    dim_risk_text = {
        "growth":         "slowing or declining revenue growth",
        "profitability":  "margin compression or low returns on capital",
        "balance_sheet":  "elevated debt load and thin liquidity",
        "fcf":            "weak free cash flow relative to reported earnings",
        "momentum":       "price underperformance and deteriorating technicals",
        "market_signals": "negative institutional positioning",
    }
    debt_vals = _last_n(metrics.get("total_debt", []))
    rev_vals = _last_n(metrics.get("revenue", []))
    high_leverage = (
        bool(debt_vals and rev_vals and rev_vals[-1] > 0)
        and (debt_vals[-1] / rev_vals[-1]) > 1.5
    )
    weakest_key, weakest_val = sorted_dims[-1]
    if high_leverage and weakest_key != "balance_sheet":
        risks = (
            f"Elevated leverage is a structural concern, compounded by "
            f"{dim_risk_text.get(weakest_key, weakest_key)} "
            f"(dimension score: {weakest_val:.0f}/100)."
        )
    else:
        risks = (
            f"The biggest risk to the thesis is "
            f"{dim_risk_text.get(weakest_key, weakest_key)} "
            f"(dimension score: {weakest_val:.0f}/100)."
        )

    # --- Verdict (ties score to standout strength and key weakness) ---
    dim_label_map = {
        "growth": "Growth Quality", "profitability": "Profitability",
        "balance_sheet": "Balance Sheet", "fcf": "FCF Quality",
        "momentum": "Momentum", "market_signals": "Market Signals",
    }
    standout_key, standout_val = sorted_dims[0]
    verdict = (
        f"{ticker}'s {composite:.1f} composite score reflects "
        f"{dim_positive[standout_key]} "
        f"({dim_label_map[standout_key]}: {standout_val:.0f}) but the "
        f"{dim_label_map[weakest_key]} score of {weakest_val:.0f} flags "
        f"{dim_risk_text.get(weakest_key, 'execution risk')} — "
        f"overall rating: {rating}."
    )

    return {
        "description": description,
        "tailwinds":   tailwinds,
        "risks":       risks,
        "verdict":     verdict,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def score_stock(
    ticker: str,
    edgar_data: dict[str, Any],
    market_data: dict[str, Any],
) -> dict[str, Any]:
    """Produce the full AlphaLens analysis for a single stock.

    Parameters
    ----------
    ticker : str
    edgar_data : dict from EdgarClient.get_financials
    market_data : dict from PolygonClient.get_market_data

    Returns
    -------
    dict with keys: ticker, company_name, composite, rating, confidence,
         dimensions, forecast, risk, summary, revenue_series, …
    """
    metrics: dict = edgar_data.get("metrics", {})

    # --- Dimension scores ---
    dimensions = {
        "growth":         round(_score_growth(metrics), 1),
        "profitability":  round(_score_profitability(metrics), 1),
        "balance_sheet":  round(_score_balance_sheet(metrics), 1),
        "fcf":            round(_score_fcf(metrics), 1),
        "momentum":       round(_score_momentum(market_data), 1),
        "market_signals": round(_score_market_signals(market_data), 1),
    }

    # --- Composite ---
    composite = sum(dimensions[k] * WEIGHTS[k] for k in WEIGHTS)
    composite = _clamp(round(composite, 1))
    rating = _rating_label(composite)

    # --- Confidence ---
    confidence = round(_compute_confidence(metrics, market_data), 1)

    # --- Forecast ---
    forecast = _compute_forecast(composite, metrics, market_data)

    # --- Risk ---
    risk = round(_compute_risk(composite, metrics), 1)

    # --- Revenue sparkline data ---
    rev_series = [
        {"end": pt["end"], "val": pt["val"]}
        for pt in metrics.get("revenue", [])[-12:]
    ]

    # --- FCF series ---
    ocf = metrics.get("operating_cash_flow", [])
    capex = metrics.get("capex", [])
    min_len = min(len(ocf), len(capex))
    fcf_series = []
    if min_len > 0:
        ocf_tail = ocf[-min_len:]
        capex_tail = capex[-min_len:]
        fcf_series = [
            {"end": ocf_tail[i]["end"], "val": ocf_tail[i]["val"] - capex_tail[i]["val"]}
            for i in range(min_len)
        ]

    # --- AI Summary ---
    summary = _generate_summary(ticker, dimensions, metrics, market_data, composite, rating)

    # --- Company Overview ---
    overview = _generate_overview(ticker, dimensions, metrics, market_data, composite, rating)

    return {
        "ticker": ticker,
        "company_name": market_data.get("company_name", ticker),
        "composite": composite,
        "rating": rating,
        "confidence": confidence,
        "dimensions": dimensions,
        "forecast": forecast,
        "risk": risk,
        "summary": summary,
        "overview": overview,
        "revenue_series": rev_series,
        "fcf_series": fcf_series,
        "market_cap": market_data.get("market_cap"),
        "last_close": market_data.get("last_close"),
        "momentum": market_data.get("momentum", "neutral"),
        "insider_activity": market_data.get("insider_activity", "unknown"),
        "options_sentiment": market_data.get("options_sentiment", "neutral"),
        "quarters_available": edgar_data.get("quarters", 0),
        # Data provenance fields
        "edgar_lag_days": edgar_data.get("edgar_lag_days", -1),
        "price_source": market_data.get("price_source", "unknown"),
        "insider_source": market_data.get("insider_source", "unknown"),
        "options_source": market_data.get("options_source", "unknown"),
        "market_data_stub": market_data.get("stub", True),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Standalone smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Synthetic test data — 8 quarters of rising revenue
    fake_metrics: dict[str, list[dict]] = {
        "revenue":            [{"end": f"2024-Q{i}", "val": 80_000 + i * 5_000, "filed": "2024-01-01", "form": "10-Q"} for i in range(8)],
        "gross_profit":       [{"end": f"2024-Q{i}", "val": 40_000 + i * 2_500, "filed": "2024-01-01", "form": "10-Q"} for i in range(8)],
        "operating_income":   [{"end": f"2024-Q{i}", "val": 20_000 + i * 1_500, "filed": "2024-01-01", "form": "10-Q"} for i in range(8)],
        "net_income":         [{"end": f"2024-Q{i}", "val": 15_000 + i * 1_200, "filed": "2024-01-01", "form": "10-Q"} for i in range(8)],
        "operating_cash_flow":[{"end": f"2024-Q{i}", "val": 18_000 + i * 1_000, "filed": "2024-01-01", "form": "10-Q"} for i in range(8)],
        "capex":              [{"end": f"2024-Q{i}", "val": 3_000 + i * 200,    "filed": "2024-01-01", "form": "10-Q"} for i in range(8)],
        "total_debt":         [{"end": f"2024-Q{i}", "val": 50_000,             "filed": "2024-01-01", "form": "10-Q"} for i in range(8)],
        "cash":               [{"end": f"2024-Q{i}", "val": 30_000 + i * 1_000, "filed": "2024-01-01", "form": "10-Q"} for i in range(8)],
        "total_assets":       [{"end": f"2024-Q{i}", "val": 200_000,            "filed": "2024-01-01", "form": "10-Q"} for i in range(8)],
        "interest_expense":   [{"end": f"2024-Q{i}", "val": 2_000,              "filed": "2024-01-01", "form": "10-Q"} for i in range(8)],
        "eps_diluted":        [{"end": f"2024-Q{i}", "val": 1.5 + i * 0.1,      "filed": "2024-01-01", "form": "10-Q"} for i in range(8)],
    }

    fake_edgar = {"ticker": "TEST", "cik": "0000000000", "metrics": fake_metrics, "quarters": 8}
    fake_market = {
        "ticker": "TEST",
        "company_name": "Test Corp",
        "market_cap": 500_000_000,
        "last_close": 150.0,
        "sma_50": 145.0,
        "sma_200": 130.0,
        "rsi_14": 58.0,
        "price_vs_sma50": 0.034,
        "price_vs_sma200": 0.154,
        "momentum": "bullish",
        "options_sentiment": "bullish",
        "insider_activity": "net_buying",
        "insider_net_shares": 5000,
        "daily_prices": [],
        "stub": False,
    }

    result = score_stock("TEST", fake_edgar, fake_market)

    print("\n" + "=" * 60)
    print("  AlphaLens AI — Scorer Smoke Test")
    print("=" * 60)
    print(f"  Ticker:      {result['ticker']}")
    print(f"  Composite:   {result['composite']}")
    print(f"  Rating:      {result['rating']}")
    print(f"  Confidence:  {result['confidence']}")
    print(f"  Risk:        {result['risk']}")
    print(f"  Forecast:    {result['forecast']}")
    print(f"  Dimensions:  {result['dimensions']}")
    print(f"\n  Summary:\n  {result['summary']}")
    print("=" * 60 + "\n")
