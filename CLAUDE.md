# AlphaLens AI — Project Guide for Claude Code

## What This Is

An async equity analysis scanner that scores stocks using SEC EDGAR fundamentals + Polygon.io market data. It produces a 0–100 composite rating across 6 dimensions (growth, profitability, balance sheet, FCF, momentum, market signals).

## Architecture

```
dashboard/index.html   ← React 18 SPA (CDN, no build step)
        ↓ HTTP
api/server.py          ← FastAPI on port 8000 (or Vercel serverless via api/index.py)
        ↓
pipeline.py            ← Orchestrates parallel fetches + scoring
   ├── clients/edgar_client.py   ← SEC EDGAR XBRL API (financials, filings)
   ├── clients/polygon_client.py ← Polygon.io (price, technicals, options, insiders)
   └── engine/scorer.py          ← 6-dimension weighted scoring engine (numpy)
```

## Running Locally

```bash
pip install -r requirements.txt
python api/server.py          # starts on http://localhost:8000
```

Or on Windows: double-click `run.bat`.

## Environment Variables

- `SEC_EDGAR_EMAIL` — **Required.** SEC EDGAR needs a valid email in the User-Agent header.
- `POLYGON_API_KEY` — Optional. Without it, market data uses stub values (fundamentals-only mode).

Set these in a `.env` file at the project root (git-ignored).

## Key API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serves the dashboard HTML |
| GET | `/health` | Liveness check |
| GET | `/ticker/{symbol}` | Score a single stock on-demand |
| POST | `/scan/start?ticker=SYM` | Background scan (optional single ticker) |
| GET | `/scan/status` | Check if a scan is running |
| GET | `/scan/results` | All results from last scan |
| GET | `/scan/results/top` | Results grouped into 4 investment lanes |
| GET | `/sources` | Metadata about all data sources |

## Scoring Engine (engine/scorer.py)

Six dimensions with these weights:
- Growth: 28% — revenue CAGR, stability, net income growth
- Profitability: 22% — gross margin, operating margin, ROIC
- FCF: 20% — FCF CAGR, FCF/NI conversion, consistency
- Balance Sheet: 15% — net debt/revenue, interest coverage, cash ratio
- Momentum: 8% — price vs SMA50/SMA200, RSI-14
- Market Signals: 7% — options sentiment, insider activity

Rating thresholds: STRONG BUY ≥78, BUY ≥63, NEUTRAL ≥48, CAUTION ≥33, RISK <33.

## Caching

All caches live in `.cache/` locally or `/tmp/.alphalens_cache` on Vercel (read-only FS).
- `{TICKER}_edgar.json` — 6-hour TTL, invalidated when new SEC accession number detected
- `{TICKER}_polygon.json` — 6-hour TTL
- `_ticker_cik_map.json` — 6-hour TTL, maps tickers to SEC CIK numbers
- `_last_scan.json` — persists last full scan results, loaded on server startup

## Deployment

**Vercel** — Live at https://alphalens-kappa.vercel.app
- `vercel.json` routes API calls to `api/index.py` (serverless Python), dashboard is static
- Env var `SEC_EDGAR_EMAIL` set via `npx vercel env add`

**GitHub** — https://github.com/dfacada/AlphaLens-AI (branch: `main`)

## Graceful Degradation

- No Polygon key → stub market data, fundamentals-only scoring
- SEC unreachable → synthetic sample data from hardcoded profiles
- API server down → dashboard shows mock stock data for development
- Read-only filesystem → cache falls back to `/tmp`

## Dashboard (dashboard/index.html)

Self-contained React 18 app using Babel CDN for JSX (no build tooling). Key features:
- Ticker input in the nav bar for single-stock analysis
- 4 investment lanes: Top AI Rated, High Growth, Short Squeeze Candidates, Undervalued Fundamentals
- Click any stock card to open a detail panel with dimension breakdowns, forecasts, and data source attribution
- Dark theme with CSS variables (`--accent: #00e5a0`)

The API base URL auto-detects: `localhost:8000` for local dev, relative paths on Vercel.

## Conventions

- All HTTP is async via `aiohttp` with `asyncio.Semaphore(5)` concurrency limit
- SEC rate limiting: ≥150ms between requests
- Retries: 3 attempts with exponential backoff on all API calls
- Data provenance tracked per result: `price_source`, `insider_source`, `options_source`, `edgar_lag_days`
- Default scan universe is in `pipeline.py` (`DEFAULT_UNIVERSE`)
- Fallback CIK mappings for 20 major stocks hardcoded in `edgar_client.py`
