#!/usr/bin/env python3
"""
Moonshot Kimi K1.5 Serving Gateway
High-concurrency FastAPI gateway with prefix caching session routing.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from kimi_long_context_engine import HierarchicalKVCache

app = FastAPI(title="Moonshot Kimi K1.5 Long-Context Gateway")
cache_manager = HierarchicalKVCache(max_context_tokens=2000000)

class LongContextRequest(BaseModel):
    session_id: str
    tokens: List[int]
    query: str
    reasoning_budget_tokens: Optional[int] = 128000

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "engine": "kimi-k1.5",
        "max_context": 2000000,
        "cache_strategy": "hierarchical_nvme_vram"
    }

@app.post("/v1/context/query")
def query_long_context(req: LongContextRequest):
    if len(req.tokens) > 2000000:
        raise HTTPException(status_code=400, detail="Token count exceeds 2,000,000 max context limit.")
    
    chunk_hash = cache_manager.compute_prefix_hash(req.tokens[:4096])
    result = cache_manager.needle_in_haystack_query(None, req.query)
    
    return {
        "session_id": req.session_id,
        "token_count": len(req.tokens),
        "prefix_cache_hit": chunk_hash in cache_manager.prefix_table,
        "retrieval_metrics": result
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
