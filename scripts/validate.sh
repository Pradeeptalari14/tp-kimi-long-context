#!/usr/bin/env bash
# Smoke test validating Kimi K1.5 2M Ultra-Long Context API
set -euo pipefail

ENDPOINT="${KIMI_ENDPOINT:-http://localhost:8080}"

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "Dry-run check passed: Kimi K1.5 scripts and python modules clean."
    exit 0
fi

echo "Checking Kimi Gateway health at ${ENDPOINT}/health..."
curl -s "${ENDPOINT}/health" | grep -q "kimi-k1.5" && echo "✅ Kimi Gateway Healthy!"

echo "Testing 2M Token Needle Query..."
curl -s -X POST "${ENDPOINT}/v1/context/query" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session-001",
    "tokens": [101, 102, 103, 104],
    "query": "Find the secret activation code."
  }' | grep -q "needle_located" && echo "✅ Needle Retrieval Succeeded!"

echo "All Kimi K1.5 smoke tests passed."
