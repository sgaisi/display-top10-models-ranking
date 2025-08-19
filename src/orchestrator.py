import time
import logging
from typing import Dict, List
from tqdm import tqdm

from app_types import CatalogEntry, RunRecord, BedrockOnDemandNotSupported
from utils import fuzzy_match_model
from providers.base import Provider

log = logging.getLogger("orchestrator")


class MultiProviderOrchestrator:
    def __init__(self, providers: List[Provider], max_new_tokens: int = 128):
        self.providers = providers
        self.max_new_tokens = max_new_tokens

    def discover(self) -> Dict[str, List[str]]:
        available = {}
        for p in self.providers:
            if p.is_ready():
                models = list(p.list_models())
                available[p.name] = models
                log.info(f"{p.name}: {len(models)} models visible")
            else:
                available[p.name] = []
                log.info(f"{p.name}: not configured")
        return available

    def match_catalog(
        self, requested_models: List[str], available: Dict[str, List[str]]
    ) -> List[CatalogEntry]:
        matched: List[CatalogEntry] = []
        for req in requested_models:
            found = False
            for prov_name, model_list in available.items():
                match = fuzzy_match_model(req, model_list)
                if match:
                    matched.append(
                        CatalogEntry(
                            canonical_name=req,
                            provider=prov_name,
                            provider_model_id=match,
                            meta={"family": prov_name},
                        )
                    )
                    found = True
                    break
            if not found:
                log.warning(f"Model '{req}' not found in any provider.")
        return matched

    def run_all(
        self, entries: List[CatalogEntry], queries: List[str]
    ) -> List[RunRecord]:
        results: List[RunRecord] = []
        prov_map: Dict[str, Provider] = {p.name: p for p in self.providers}

        total = len(entries) * len(queries)
        with tqdm(total=total, desc="Running matched models", ncols=100) as pbar:
            for e in entries:
                prov = prov_map[e.provider]
                for q in queries:
                    log.info(
                        f"""[{e.provider}] 
                             {e.provider_model_id} :: {q[:48]}…"""
                    )
                    try:
                        text, rt = prov.run(q, e.provider_model_id)
                        preview = (
                            (text or "").strip().replace("\r", " ").replace("\n", " ")
                        )
                        if len(preview) > 240:
                            preview = preview[:240] + "…"
                        rec = RunRecord(
                            provider=e.provider,
                            model_id=e.provider_model_id,
                            canonical_name=e.canonical_name,
                            query=q,
                            ok=True,
                            response_time=rt,
                            response_preview=preview,
                        )
                        log.info(
                            f"{e.provider}:{e.canonical_name} [{rt:.2f}s] {preview}"
                        )
                    except BedrockOnDemandNotSupported as ex:
                        rec = RunRecord(
                            provider=e.provider,
                            model_id=e.provider_model_id,
                            canonical_name=e.canonical_name,
                            query=q,
                            ok=False,
                            error=str(ex),
                        )
                        log.warning(f"{e.provider}:{e.canonical_name} ERROR: {ex}")
                    except Exception as ex:
                        rec = RunRecord(
                            provider=e.provider,
                            model_id=e.provider_model_id,
                            canonical_name=e.canonical_name,
                            query=q,
                            ok=False,
                            error=str(ex),
                        )
                        log.warning(f"{e.provider}:{e.canonical_name} ERROR: {ex}")
                    results.append(rec)
                    pbar.update(1)
                    time.sleep(0.25)
        return results
