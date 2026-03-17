"""
AlphaLens AI – Async Data Pipeline
====================================
Orchestrates the full scan workflow:
  1. Fetch EDGAR financial data for each ticker.
  2. Fetch Polygon market data for each ticker.
  3. Run the scoring engine.
  4. Aggregate and rank results.

Supports parallel async scanning of the full ticker universe.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from clients.edgar_client import EdgarClient
from clients.polygon_client import PolygonClient
from engine.scorer import score_stock

logger = logging.getLogger("alphalens.pipeline")


# ---------------------------------------------------------------------------
# Default scan universe
# ---------------------------------------------------------------------------
DEFAULT_UNIVERSE: list[str] = [
    "NVDA",
]


# ---------------------------------------------------------------------------
# Single-ticker pipeline
# ---------------------------------------------------------------------------
async def _scan_one(
    ticker: str,
    edgar: EdgarClient,
    polygon: PolygonClient,
    session: aiohttp.ClientSession,
) -> dict[str, Any] | None:
    """Fetch data and score a single ticker. Returns None on hard failure."""
    try:
        edgar_data, market_data = await asyncio.gather(
            edgar.get_financials(ticker, session=session),
            polygon.get_market_data(ticker, session=session),
        )

        result = score_stock(ticker, edgar_data, market_data)
        logger.info(
            "Scored %s → %s (%.1f)",
            ticker, result["rating"], result["composite"],
        )
        return result

    except Exception:
        logger.exception("Pipeline failed for %s", ticker)
        return None


# ---------------------------------------------------------------------------
# Full scan
# ---------------------------------------------------------------------------
async def run_scan(
    tickers: list[str] | None = None,
    concurrency: int = 5,
) -> list[dict[str, Any]]:
    """Scan *tickers* (default universe) with bounded concurrency.

    Returns a list of scored results sorted by composite score (descending).
    """
    tickers = tickers or DEFAULT_UNIVERSE
    edgar = EdgarClient()
    polygon = PolygonClient()
    results: list[dict[str, Any]] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def _guarded(t: str, sess: aiohttp.ClientSession) -> None:
        async with semaphore:
            res = await _scan_one(t, edgar, polygon, sess)
            if res is not None:
                results.append(res)

    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_guarded(t, session) for t in tickers]
        await asyncio.gather(*tasks)

    # Sort by composite score descending
    results.sort(key=lambda r: r.get("composite", 0), reverse=True)
    logger.info("Scan complete — %d/%d tickers scored", len(results), len(tickers))
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    async def _main() -> None:
        results = await run_scan()
        for r in results:
            print(f"  {r['ticker']:6s}  {r['composite']:5.1f}  {r['rating']}")
        # Also dump full JSON for inspection
        with open(".cache/_last_scan.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nFull results written to .cache/_last_scan.json")

    asyncio.run(_main())
