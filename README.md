# AlphaLens AI — Equity Analysis Scanner

A local-first equity analysis scanner that combines SEC EDGAR XBRL filings with Polygon.io market data to produce AI-powered fundamental ratings for stocks.

## Architecture

```
alphalens/
├── .env.example          # Configuration template
├── requirements.txt      # Python dependencies
├── pipeline.py           # Async data pipeline (orchestrates fetching + scoring)
├── clients/
│   ├── edgar_client.py   # SEC EDGAR XBRL API client
│   └── polygon_client.py # Polygon.io market data client
├── engine/
│   └── scorer.py         # 6-dimension scoring engine (0–100)
├── api/
│   └── server.py         # FastAPI REST API
└── dashboard/
    └── index.html        # Self-contained React dashboard (no build step)
```

## Scoring System

Each stock receives a composite score (0–100) built from six weighted dimensions:

| Dimension | Weight | Key Metrics |
|-----------|--------|-------------|
| Growth Quality | 28% | Revenue CAGR, revenue stability, net income growth |
| Profitability | 22% | Gross margin, operating margin, ROIC |
| Free Cash Flow | 20% | FCF CAGR, FCF/NI conversion, consistency |
| Balance Sheet | 15% | Net debt/revenue, interest coverage, cash/assets |
| Momentum | 8% | Price vs SMA50/200, RSI |
| Market Signals | 7% | Options sentiment, insider activity |

**Rating Scale:** STRONG BUY (≥78) → BUY (≥63) → NEUTRAL (≥48) → CAUTION (≥33) → RISK (<33)

## Quick Start

```bash
# 1. Configure
cp .env.example .env
# Edit .env with your email (required for SEC) and optionally a Polygon API key

# 2. Install
pip install -r requirements.txt

# 3. Smoke test the scorer
python engine/scorer.py

# 4. Start API server
python api/server.py

# 5. Trigger a scan (in another terminal)
curl -X POST http://localhost:8000/scan/start

# 6. View results
curl http://localhost:8000/scan/results/top

# 7. Open the dashboard
open dashboard/index.html
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/ticker/{symbol}` | Score a single ticker |
| POST | `/scan/start` | Start background universe scan |
| GET | `/scan/status` | Check scan progress |
| GET | `/scan/results` | All results from last scan |
| GET | `/scan/results/top` | Results grouped into investment lanes |

## Data Sources

- **SEC EDGAR** (free, no API key): Company financial statements via XBRL API
- **Polygon.io** (optional): Price data, technicals, options flow, insider transactions

The system gracefully degrades without a Polygon key — scores are computed from fundamentals only.

## Default Scan Universe

AAPL, MSFT, NVDA, META, GOOGL, AMZN, JPM, V, UNH, JNJ, XOM, PG, HD, MA, AVGO, MRK, COST, CVX, ABBV, PEP

## Tech Stack

- Python 3.10+ with async/await throughout
- FastAPI + Uvicorn
- aiohttp for all HTTP
- pandas / numpy / scipy for computation
- React 18 (CDN, no build step)
