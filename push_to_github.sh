#!/bin/bash
# ─────────────────────────────────────────────────
# AlphaLens AI — Push to GitHub
# Run this from your local machine inside the alphalens/ folder
# ─────────────────────────────────────────────────

set -e

REPO_NAME="alphalens-ai"

echo "Creating private GitHub repo: $REPO_NAME"
gh repo create "$REPO_NAME" --private --source=. --push --description "AI-powered equity analysis scanner — SEC EDGAR + Polygon.io"

echo ""
echo "Done! Your repo is live at:"
gh repo view --web
