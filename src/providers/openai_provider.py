import os
import time
from typing import Iterable, Tuple
from .base import Provider

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self):
        self.client = None
        if OpenAI and os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                organization=os.getenv("OPENAI_ORG_ID"),
                project=os.getenv("OPENAI_PROJECT"),
            )

    def is_ready(self) -> bool:
        return self.client is not None

    def list_models(self) -> Iterable[str]:
        if not self.is_ready():
            return []
        try:
            models = self.client.models.list()
            for m in models.data if hasattr(models, "data") else models:
                yield (m.id if hasattr(m, "id") else str(m)).lower()
        except Exception:
            return []

    def run(self, prompt: str, model_id: str) -> Tuple[str, float]:
        if not self.is_ready():
            raise RuntimeError("OpenAI client not initialized")
        t0 = time.time()
        out = self.client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.7,
        )
        txt = out.choices[0].message.content or ""
        return txt, time.time() - t0
