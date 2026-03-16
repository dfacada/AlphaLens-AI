"""
AlphaLens AI – Polygon.io Market Data Client
=============================================
Fetches price data, technical indicators, options sentiment, and insider
transactions from the Polygon.io REST API.

If no API key is configured the client returns **stubbed placeholder data**
so the scanner can still run in fundamentals-only mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("alphalens.polygon")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_SECONDS = 6 * 3600  # 6 hours

POLYGON_BASE = "https://api.polygon.io"
API_KEY = os.getenv("POLYGON_API_KEY", "")

MAX_RETRIES = 3
BACKOFF_BASE = 1.0
REQUEST_INTERVAL = 0.15  # rate-limit courtesy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}_polygon.json"


def _cache_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size < 20:
        return False
    return (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS


class _RateLimiter:
    def __init__(self, interval: float = REQUEST_INTERVAL):
        self._interval = interval
        self._last: float = 0.0

    async def wait(self) -> None:
        now = time.monotonic()
        gap = self._interval - (now - self._last)
        if gap > 0:
            await asyncio.sleep(gap)
        self._last = time.monotonic()


_limiter = _RateLimiter()


async def _fetch_json(
    session: aiohttp.ClientSession, url: str, params: dict[str, str] | None = None
) -> dict[str, Any] | None:
    """GET *url* with Polygon auth, retry, and rate-limiting."""
    if params is None:
        params = {}
    params["apiKey"] = API_KEY

    for attempt in range(1, MAX_RETRIES + 1):
        await _limiter.wait()
        try:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    return await resp.json(content_type=None)
                if resp.status == 429 or resp.status >= 500:
                    logger.warning("Polygon %s → %s (attempt %d)", url, resp.status, attempt)
                else:
                    logger.warning("Polygon %s → %s — not retrying", url, resp.status)
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("Polygon request error: %s (attempt %d)", exc, attempt)

        if attempt < MAX_RETRIES:
            await asyncio.sleep(BACKOFF_BASE * (2 ** (attempt - 1)))

    return None


# ---------------------------------------------------------------------------
# Stub / fallback data
# ---------------------------------------------------------------------------
_STUB_PROFILES: dict[str, dict[str, Any]] = {
    "AAPL":  {"name":"Apple Inc.",            "cap":3200e9,  "price":228, "sma50":222, "sma200":205, "rsi":58, "opt":"bullish",  "ins":"neutral"},
    "MSFT":  {"name":"Microsoft Corp.",       "cap":3100e9,  "price":445, "sma50":438, "sma200":410, "rsi":60, "opt":"bullish",  "ins":"net_buying"},
    "NVDA":  {"name":"NVIDIA Corp.",          "cap":2900e9,  "price":135, "sma50":128, "sma200":100, "rsi":65, "opt":"bullish",  "ins":"neutral"},
    "META":  {"name":"Meta Platforms Inc.",   "cap":1500e9,  "price":580, "sma50":565, "sma200":510, "rsi":55, "opt":"neutral",  "ins":"neutral"},
    "GOOGL": {"name":"Alphabet Inc.",         "cap":2100e9,  "price":175, "sma50":172, "sma200":162, "rsi":52, "opt":"neutral",  "ins":"neutral"},
    "AMZN":  {"name":"Amazon.com Inc.",       "cap":2000e9,  "price":210, "sma50":205, "sma200":190, "rsi":54, "opt":"bullish",  "ins":"neutral"},
    "JPM":   {"name":"JPMorgan Chase & Co.",  "cap":660e9,   "price":230, "sma50":225, "sma200":210, "rsi":56, "opt":"neutral",  "ins":"net_buying"},
    "V":     {"name":"Visa Inc.",             "cap":590e9,   "price":310, "sma50":305, "sma200":285, "rsi":53, "opt":"neutral",  "ins":"neutral"},
    "UNH":   {"name":"UnitedHealth Group",    "cap":480e9,   "price":520, "sma50":535, "sma200":540, "rsi":42, "opt":"bearish",  "ins":"net_selling"},
    "JNJ":   {"name":"Johnson & Johnson",     "cap":370e9,   "price":155, "sma50":158, "sma200":160, "rsi":44, "opt":"neutral",  "ins":"neutral"},
    "XOM":   {"name":"Exxon Mobil Corp.",     "cap":480e9,   "price":112, "sma50":115, "sma200":110, "rsi":48, "opt":"neutral",  "ins":"net_buying"},
    "PG":    {"name":"Procter & Gamble Co.",  "cap":380e9,   "price":165, "sma50":162, "sma200":158, "rsi":51, "opt":"neutral",  "ins":"neutral"},
    "HD":    {"name":"Home Depot Inc.",       "cap":370e9,   "price":380, "sma50":375, "sma200":360, "rsi":50, "opt":"neutral",  "ins":"neutral"},
    "MA":    {"name":"Mastercard Inc.",       "cap":440e9,   "price":490, "sma50":480, "sma200":460, "rsi":57, "opt":"bullish",  "ins":"neutral"},
    "AVGO":  {"name":"Broadcom Inc.",         "cap":800e9,   "price":195, "sma50":185, "sma200":155, "rsi":62, "opt":"bullish",  "ins":"neutral"},
    "MRK":   {"name":"Merck & Co. Inc.",      "cap":300e9,   "price":118, "sma50":122, "sma200":125, "rsi":40, "opt":"bearish",  "ins":"neutral"},
    "COST":  {"name":"Costco Wholesale",      "cap":400e9,   "price":920, "sma50":900, "sma200":850, "rsi":58, "opt":"bullish",  "ins":"neutral"},
    "CVX":   {"name":"Chevron Corp.",         "cap":290e9,   "price":155, "sma50":158, "sma200":155, "rsi":47, "opt":"neutral",  "ins":"net_buying"},
    "ABBV":  {"name":"AbbVie Inc.",           "cap":310e9,   "price":178, "sma50":175, "sma200":170, "rsi":53, "opt":"neutral",  "ins":"neutral"},
    "PEP":   {"name":"PepsiCo Inc.",          "cap":220e9,   "price":162, "sma50":165, "sma200":170, "rsi":41, "opt":"neutral",  "ins":"neutral"},
}


def _stub_data(ticker: str) -> dict[str, Any]:
    """Return realistic placeholder data when no Polygon key is configured."""
    p = _STUB_PROFILES.get(ticker, {})
    price = p.get("price", 100)
    sma50 = p.get("sma50", 100)
    sma200 = p.get("sma200", 100)
    pvs50 = (price - sma50) / sma50 if sma50 else 0
    pvs200 = (price - sma200) / sma200 if sma200 else 0

    momentum = "neutral"
    if pvs50 > 0.03 and pvs200 > 0.05:
        momentum = "bullish"
    elif pvs50 < -0.03 and pvs200 < -0.05:
        momentum = "bearish"

    return {
        "ticker": ticker,
        "company_name": p.get("name", ticker),
        "market_cap": p.get("cap"),
        "last_close": price,
        "sma_50": sma50,
        "sma_200": sma200,
        "rsi_14": p.get("rsi", 50),
        "price_vs_sma50": round(pvs50, 4),
        "price_vs_sma200": round(pvs200, 4),
        "momentum": momentum,
        "options_sentiment": p.get("opt", "neutral"),
        "insider_activity": p.get("ins", "unknown"),
        "insider_net_shares": 0,
        "daily_prices": [],
        "stub": True,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class PolygonClient:
    """Async Polygon.io client with graceful degradation."""

    async def get_market_data(
        self, ticker: str, *, session: aiohttp.ClientSession
    ) -> dict[str, Any]:
        ticker = ticker.upper()
        cache = _cache_path(ticker)

        # 1. Check cache
        if _cache_is_valid(cache):
            logger.info("Polygon cache hit for %s", ticker)
            return json.loads(cache.read_text())

        # 2. No API key → return stubs
        if not API_KEY:
            logger.info("No Polygon API key — returning stub data for %s", ticker)
            stub = _stub_data(ticker)
            cache.write_text(json.dumps(stub))
            return stub

        # 3. Fetch all data points in parallel
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        start_date = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")

        ticker_url = f"{POLYGON_BASE}/v3/reference/tickers/{ticker}"
        agg_url = f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        sma50_url = f"{POLYGON_BASE}/v1/indicators/sma/{ticker}"
        sma200_url = f"{POLYGON_BASE}/v1/indicators/sma/{ticker}"
        rsi_url = f"{POLYGON_BASE}/v1/indicators/rsi/{ticker}"
        options_url = f"{POLYGON_BASE}/v3/snapshot/options/{ticker}"
        insider_url = f"{POLYGON_BASE}/v2/reference/insider-transactions"

        (
            ticker_info,
            agg_data,
            sma50_data,
            sma200_data,
            rsi_data,
            options_data,
            insider_data,
        ) = await asyncio.gather(
            _fetch_json(session, ticker_url),
            _fetch_json(session, agg_url, {"adjusted": "true", "sort": "asc", "limit": "5000"}),
            _fetch_json(session, sma50_url, {"window": "50", "timespan": "day", "series_type": "close"}),
            _fetch_json(session, sma200_url, {"window": "200", "timespan": "day", "series_type": "close"}),
            _fetch_json(session, rsi_url, {"window": "14", "timespan": "day", "series_type": "close"}),
            _fetch_json(session, options_url),
            _fetch_json(session, insider_url, {"ticker": ticker, "limit": "50"}),
        )

        # --- Parse ticker info ---
        company_name = ticker
        market_cap = None
        if ticker_info and "results" in ticker_info:
            res = ticker_info["results"]
            company_name = res.get("name", ticker)
            market_cap = res.get("market_cap")

        # --- Parse daily prices & last close ---
        daily_prices: list[dict[str, Any]] = []
        last_close: float | None = None
        if agg_data and "results" in agg_data:
            for bar in agg_data["results"]:
                daily_prices.append({"t": bar.get("t"), "c": bar.get("c")})
            if daily_prices:
                last_close = daily_prices[-1]["c"]

        # --- SMA50 ---
        sma_50: float | None = None
        if sma50_data and "results" in sma50_data:
            vals = sma50_data["results"].get("values", [])
            if vals:
                sma_50 = vals[0].get("value")

        # --- SMA200 ---
        sma_200: float | None = None
        if sma200_data and "results" in sma200_data:
            vals = sma200_data["results"].get("values", [])
            if vals:
                sma_200 = vals[0].get("value")

        # --- RSI ---
        rsi_14: float = 50.0
        if rsi_data and "results" in rsi_data:
            vals = rsi_data["results"].get("values", [])
            if vals:
                rsi_14 = vals[0].get("value", 50.0)

        # --- Momentum labels ---
        price_vs_sma50 = 0.0
        price_vs_sma200 = 0.0
        if last_close and sma_50:
            price_vs_sma50 = (last_close - sma_50) / sma_50
        if last_close and sma_200:
            price_vs_sma200 = (last_close - sma_200) / sma_200

        momentum = "neutral"
        if price_vs_sma50 > 0.03 and price_vs_sma200 > 0.05:
            momentum = "bullish"
        elif price_vs_sma50 < -0.03 and price_vs_sma200 < -0.05:
            momentum = "bearish"

        # --- Options sentiment ---
        options_sentiment = "neutral"
        if options_data and "results" in options_data:
            results = options_data["results"]
            if isinstance(results, list) and len(results) > 0:
                total_call_oi = sum(r.get("details", {}).get("open_interest", 0) for r in results if r.get("details", {}).get("contract_type") == "call")
                total_put_oi = sum(r.get("details", {}).get("open_interest", 0) for r in results if r.get("details", {}).get("contract_type") == "put")
                if total_call_oi > total_put_oi * 1.3:
                    options_sentiment = "bullish"
                elif total_put_oi > total_call_oi * 1.3:
                    options_sentiment = "bearish"

        # --- Insider transactions ---
        insider_activity = "unknown"
        insider_net_shares = 0
        if insider_data and "results" in insider_data:
            for txn in insider_data["results"]:
                shares = txn.get("shares", 0) or 0
                acq_disp = txn.get("acquisition_or_disposition", "")
                if acq_disp == "A":
                    insider_net_shares += shares
                elif acq_disp == "D":
                    insider_net_shares -= shares
            if insider_net_shares > 1000:
                insider_activity = "net_buying"
            elif insider_net_shares < -1000:
                insider_activity = "net_selling"
            else:
                insider_activity = "neutral"

        result: dict[str, Any] = {
            "ticker": ticker,
            "company_name": company_name,
            "market_cap": market_cap,
            "last_close": last_close,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "rsi_14": rsi_14,
            "price_vs_sma50": price_vs_sma50,
            "price_vs_sma200": price_vs_sma200,
            "momentum": momentum,
            "options_sentiment": options_sentiment,
            "insider_activity": insider_activity,
            "insider_net_shares": insider_net_shares,
            "daily_prices": daily_prices[-90:],  # keep last ~90 trading days
            "stub": False,
        }

        cache.write_text(json.dumps(result, default=str))
        logger.info("Fetched & cached Polygon data for %s", ticker)
        return result
