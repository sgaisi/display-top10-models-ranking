import json
import time
import logging
from typing import Iterable, Tuple, Dict
import boto3
from .base import Provider
from app_types import BedrockOnDemandNotSupported


log = logging.getLogger("bedrock")


class BedrockProvider(Provider):
    name = "bedrock"

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        try:
            boto3.setup_default_session(region_name=self.region)
            self.client = boto3.client("bedrock", region_name=self.region)
            self.runtime = boto3.client("bedrock-runtime", region_name=self.region)
        except Exception:
            self.client = None
            self.runtime = None
        self._lookup: Dict[str, Dict[str, str]] = {}
        if self.client:
            self._load_model_mappings()

    def is_ready(self) -> bool:
        return self.client is not None and self.runtime is not None

    def _load_model_mappings(self):
        self._lookup = {}
        try:
            resp = self.client.list_foundation_models()
            for s in resp.get("modelSummaries", []):
                keys = []
                if "modelName" in s:
                    keys.append(s["modelName"].lower())
                if "modelId" in s:
                    keys.append(s["modelId"].lower())
                for k in keys:
                    self._lookup[k] = {
                        "modelId": s["modelId"],
                        "modelArn": s.get("modelArn"),
                    }
        except Exception as e:
            log.warning(f"Bedrock model mapping failed: {e}")

    def list_models(self) -> Iterable[str]:
        if not self.is_ready():
            return []
        try:
            resp = self.client.list_foundation_models()
            for s in resp.get("modelSummaries", []):
                if "modelId" in s:
                    yield s["modelId"].lower()
        except Exception:
            return []

    def run(self, prompt: str, model_id: str) -> Tuple[str, float]:
        if not self.is_ready():
            raise RuntimeError("Bedrock not initialized")
        t0 = time.time()
        resolved = self._lookup.get(
            model_id.lower(), {"modelId": model_id, "modelArn": None}
        )
        resolved_id = resolved["modelId"]
        family = resolved_id.split(".")[0]
        if family == "anthropic":
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 128,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
            }
        else:
            body = {"inputText": prompt}
        try:
            resp = self.runtime.invoke_model(
                modelId=resolved_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            raw = resp["body"].read().decode("utf-8")
            if family == "anthropic":
                try:
                    parsed = json.loads(raw)
                    msg = (
                        parsed.get("output", {}).get("message")
                        or parsed.get("content")
                        or parsed.get("messages")
                    )
                    if isinstance(msg, list):
                        parts = []
                        for m in msg:
                            if isinstance(m, dict) and "content" in m:
                                for p in m["content"]:
                                    if isinstance(p, dict) and p.get("type") == "text":
                                        parts.append(p.get("text", ""))
                        text = "\n".join(parts).strip()
                    else:
                        text = str(msg)
                except Exception:
                    text = raw
            else:
                text = raw
            return text or "", time.time() - t0
        except Exception as e:
            if "on-demand throughput isn’t supported" in str(e):
                raise BedrockOnDemandNotSupported(
                    f"On-demand not supported for {resolved_id}"
                )
            raise
