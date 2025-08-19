import os
import time
from typing import Iterable, Tuple
from .base import Provider

try:
    from together import Together
except Exception:
    Together = None


class TogetherProvider(Provider):
    name = "together"

    def __init__(self):
        self.client = (
            Together(api_key=os.getenv("TOGETHER_API_KEY"))
            if (Together and os.getenv("TOGETHER_API_KEY"))
            else None
        )

    def is_ready(self) -> bool:
        return self.client is not None

    def list_models(self) -> Iterable[str]:
        if not self.is_ready():
            return []
        try:
            models = self.client.models.list()

            def extract_id(m):
                if isinstance(m, dict) and "id" in m:
                    return str(m["id"])
                if hasattr(m, "id"):
                    return str(getattr(m, "id"))
                s = str(m)
                return s[3:].split("'", 1)[0] if s.startswith("id='") else s

            for m in models:
                yield extract_id(m).lower()
        except Exception:
            return []

    def run(self, prompt: str, model_id: str) -> Tuple[str, float]:
        if not self.is_ready():
            raise RuntimeError("Together client not initialized")
        t0 = time.time()
        out = self.client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.7,
        )
        txt = out.choices[0].message.content or ""
        return txt, time.time() - t0
