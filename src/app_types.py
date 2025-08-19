from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CatalogEntry:
    canonical_name: str
    provider: str
    provider_model_id: str
    meta: Dict[str, Any]


@dataclass
class RunRecord:
    provider: str
    model_id: str
    canonical_name: str
    query: str
    ok: bool
    response_time: Optional[float] = None
    response_preview: Optional[str] = None
    error: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class BedrockOnDemandNotSupported(Exception):
    pass
