"""Dynamic model list fetcher from Z.AI API."""
import httpx
from typing import List, Dict, Any

_MODEL_CACHE_URL = "https://chat.z.ai/api/models"
_model_cache: List[Dict[str, Any]] | None = None

async def fetch_models() -> List[Dict[str, Any]]:
    """Fetch available models from Z.AI API and cache them."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(_MODEL_CACHE_URL)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    _model_cache = data.get("models", [])
                elif isinstance(data, list):
                    _model_cache = data
                else:
                    _model_cache = []
            else:
                _model_cache = []
    except Exception:
        _model_cache = []
    return _model_cache
