"""
AlphaLens AI – FastAPI Server
==============================
Exposes the scanning pipeline and results via a REST API.

Endpoints
---------
GET  /              → redirect to dashboard
GET  /health        → liveness check
GET  /ticker/{sym}  → score a single ticker on-demand
POST /scan/start    → kick off a background full-universe scan
GET  /scan/status   → check whether a scan is running
GET  /scan/results  → return all results from the last scan
GET  /scan/results/top → top results grouped into investment lanes

Run:  python api/server.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path so relative imports work when
# the file is executed directly (python api/server.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from clients.edgar_client import EdgarClient
from clients.polygon_client import PolygonClient
from engine.scorer import score_stock
from pipeline import run_scan, DEFAULT_UNIVERSE

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("alphalens.api")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AlphaLens AI",
    version="1.0.0",
    description="Equity fundamental analysis scanner powered by SEC EDGAR & Polygon.io",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory state for scan results
# ---------------------------------------------------------------------------
_scan_state: dict[str, Any] = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "tickers_total": 0,
    "tickers_done": 0,
    "results": [],
}

RESULTS_CACHE = PROJECT_ROOT / ".cache" / "_last_scan.json"


def _load_cached_results() -> list[dict]:
    """Load previously cached scan results if they exist."""
    if RESULTS_CACHE.exists():
        try:
            return json.loads(RESULTS_CACHE.read_text())
        except Exception:
            pass
    return []


# Pre-load cached results on startup
_scan_state["results"] = _load_cached_results()


# ---------------------------------------------------------------------------
# Background scan task
# ---------------------------------------------------------------------------
async def _background_scan(tickers: list[str] | None = None) -> None:
    """Run the full pipeline in the background."""
    tickers = tickers or DEFAULT_UNIVERSE
    _scan_state["running"] = True
    _scan_state["started_at"] = time.time()
    _scan_state["tickers_total"] = len(tickers)
    _scan_state["tickers_done"] = 0

    try:
        results = await run_scan(tickers=tickers)
        _scan_state["results"] = results
        _scan_state["tickers_done"] = len(results)

        # Persist to cache
        RESULTS_CACHE.parent.mkdir(exist_ok=True)
        RESULTS_CACHE.write_text(json.dumps(results, indent=2, default=str))
        logger.info("Scan complete — %d results cached", len(results))

    except Exception:
        logger.exception("Background scan failed")
    finally:
        _scan_state["running"] = False
        _scan_state["finished_at"] = time.time()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the dashboard HTML."""
    dashboard = PROJECT_ROOT / "dashboard" / "index.html"
    if dashboard.exists():
        return FileResponse(str(dashboard), media_type="text/html")
    return JSONResponse({"message": "AlphaLens AI API is running. Dashboard not found."})


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/ticker/{symbol}")
async def single_ticker(symbol: str):
    """Score a single ticker on-demand (not part of a full scan)."""
    symbol = symbol.upper()
    edgar = EdgarClient()
    polygon = PolygonClient()

    connector = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        edgar_data, market_data = await asyncio.gather(
            edgar.get_financials(symbol, session=session),
            polygon.get_market_data(symbol, session=session),
        )

    result = score_stock(symbol, edgar_data, market_data)
    return result


@app.post("/scan/start")
async def scan_start(background_tasks: BackgroundTasks, ticker: str | None = Query(default=None)):
    """Kick off a background scan. Pass ?ticker=AAPL to scan a single stock."""
    if _scan_state["running"]:
        return {"status": "already_running", "started_at": _scan_state["started_at"]}

    tickers = [ticker.upper()] if ticker else DEFAULT_UNIVERSE
    background_tasks.add_task(_background_scan, tickers)
    return {"status": "started", "tickers": len(tickers), "universe": tickers}


@app.get("/scan/status")
async def scan_status():
    return {
        "running": _scan_state["running"],
        "started_at": _scan_state["started_at"],
        "finished_at": _scan_state["finished_at"],
        "tickers_total": _scan_state["tickers_total"],
        "tickers_done": _scan_state["tickers_done"],
    }


@app.get("/scan/results")
async def scan_results():
    """Return all results from the last completed scan."""
    return _scan_state["results"]


@app.get("/scan/results/top")
async def scan_results_top():
    """Return results grouped into the four dashboard investment lanes."""
    results = _scan_state["results"]
    if not results:
        return {"lanes": [], "total": 0}

    # Sort copies for each lane
    by_composite = sorted(results, key=lambda r: r.get("composite", 0), reverse=True)
    by_growth = sorted(results, key=lambda r: r.get("dimensions", {}).get("growth", 0), reverse=True)
    by_fcf_value = sorted(
        results,
        key=lambda r: (
            r.get("dimensions", {}).get("fcf", 0)
            + r.get("dimensions", {}).get("balance_sheet", 0)
        ),
        reverse=True,
    )
    # "Short Squeeze Candidates" — low composite but high momentum/market signals
    by_squeeze = sorted(
        results,
        key=lambda r: (
            r.get("dimensions", {}).get("momentum", 0)
            + r.get("dimensions", {}).get("market_signals", 0)
            - r.get("composite", 50) * 0.3
        ),
        reverse=True,
    )

    lanes = [
        {"title": "Top AI Rated",               "stocks": by_composite[:8]},
        {"title": "High Growth",                 "stocks": by_growth[:8]},
        {"title": "Short Squeeze Candidates",    "stocks": by_squeeze[:8]},
        {"title": "Undervalued Fundamentals",    "stocks": by_fcf_value[:8]},
    ]

    return {"lanes": lanes, "total": len(results)}


@app.get("/sources")
async def data_sources():
    """Return metadata about each data source used by AlphaLens."""
    from clients.edgar_client import CACHE_TTL_SECONDS as EDGAR_TTL
    from clients.polygon_client import CACHE_TTL_SECONDS as POLYGON_TTL, API_KEY as POLY_KEY

    return {
        "sources": [
            {
                "name": "SEC EDGAR (XBRL)",
                "endpoint": "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                "type": "filing-based",
                "description": "10-Q and 10-K financial statements",
                "cache_ttl_seconds": EDGAR_TTL,
                "notes": "Cache invalidated when new accession number detected",
            },
            {
                "name": "SEC EDGAR (Submissions)",
                "endpoint": "https://data.sec.gov/submissions/CIK{cik}.json",
                "type": "filing-based",
                "description": "Detects new filings to invalidate stale caches",
            },
            {
                "name": "Polygon.io Snapshot",
                "endpoint": "/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}",
                "type": "real-time",
                "description": "Current day price for momentum calculations",
                "cache_ttl_seconds": POLYGON_TTL,
                "available": bool(POLY_KEY),
            },
            {
                "name": "Polygon.io Aggregates",
                "endpoint": "/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
                "type": "delayed",
                "description": "Daily OHLCV bars (fallback for price)",
                "available": bool(POLY_KEY),
            },
            {
                "name": "Polygon.io Technical Indicators",
                "endpoint": "/v1/indicators/sma|rsi/{ticker}",
                "type": "delayed",
                "description": "SMA-50, SMA-200, RSI-14",
                "available": bool(POLY_KEY),
            },
            {
                "name": "Polygon.io Options Snapshot",
                "endpoint": "/v3/snapshot/options/{ticker}",
                "type": "delayed",
                "description": "Options open interest for sentiment",
                "available": bool(POLY_KEY),
            },
            {
                "name": "Polygon.io Insider Transactions",
                "endpoint": "/v2/reference/insider-transactions?ticker={ticker}",
                "type": "delayed",
                "description": "Insider buy/sell activity",
                "available": bool(POLY_KEY),
                "fallback": "SEC EDGAR Form 4 via EFTS",
            },
            {
                "name": "SEC EDGAR Form 4 (Fallback)",
                "endpoint": "https://efts.sec.gov/LATEST/search-index?forms=4",
                "type": "filing-based",
                "description": "Insider filings fallback when Polygon data is empty",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
