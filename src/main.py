import os
import json
import logging
from pathlib import Path

from orchestrator import MultiProviderOrchestrator
from providers.openai_provider import OpenAIProvider
from providers.together_provider import TogetherProvider
from providers.gemini_provider import GeminiProvider
from providers.bedrock_provider import BedrockProvider
from providers.jumpstart_provider import JumpStartProvider

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("main")

ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_JSON = ROOT / "data" / "top_10_models.json"


def load_requested_models(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # expect either {"models": [...]} or a plain list
    return data.get("models", data if isinstance(data, list) else [])


def main():
    providers = [
        OpenAIProvider(),
        TogetherProvider(),
        GeminiProvider(),
        BedrockProvider(region=os.getenv("AWS_BEDROCK_REGION", "us-east-1")),
        JumpStartProvider(
            region=os.getenv("AWS_JUMPSTART_REGION", "us-east-2"),
            exec_role_arn=os.getenv("JUMPSTART_EXEC_ROLE_ARN"),
            default_instance=os.getenv("JUMPSTART_INSTANCE", "ml.g5.2xlarge"),
        ),
    ]

    orch = MultiProviderOrchestrator(providers, max_new_tokens=128)

    models_path = os.getenv("MODELS_JSON", str(DEFAULT_MODELS_JSON))
    requested = load_requested_models(models_path)
    available = orch.discover()
    entries = orch.match_catalog(requested, available)

    log.info("=== Top Matches ===")
    for i, e in enumerate(entries, 1):
        log.info(f"{i:2d}. [{e.provider}] {e.canonical_name} -> {e.provider_model_id}")

    queries = ["In one sentence, explain why caching improves performance."]
    results = orch.run_all(entries, queries)

    ok = sum(1 for r in results if r.ok)
    log.info("=" * 50)
    log.info(f"Done. {ok}/{len(results)} successful.")
    by_provider = {}
    for r in results:
        print(r)
        sc, tot = by_provider.get(r.provider, (0, 0))
        by_provider[r.provider] = (sc + (1 if r.ok else 0), tot + 1)
    for p, (sc, tot) in by_provider.items():
        log.info(f"{p:10s}: {sc}/{tot} ok")


if __name__ == "__main__":
    main()
