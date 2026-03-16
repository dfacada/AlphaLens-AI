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
from fastapi import FastAPI, BackgroundTasks
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
async def _background_scan() -> None:
    """Run the full pipeline in the background."""
    _scan_state["running"] = True
    _scan_state["started_at"] = time.time()
    _scan_state["tickers_total"] = len(DEFAULT_UNIVERSE)
    _scan_state["tickers_done"] = 0

    try:
        results = await run_scan()
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
async def scan_start(background_tasks: BackgroundTasks):
    """Kick off a background scan of the default universe."""
    if _scan_state["running"]:
        return {"status": "already_running", "started_at": _scan_state["started_at"]}

    background_tasks.add_task(_background_scan)
    return {"status": "started", "tickers": len(DEFAULT_UNIVERSE)}


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
