import os
import time
from typing import Iterable, Tuple
from .base import Provider

try:
    import google.generativeai as genai
except Exception:
    genai = None


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self):
        self.ready = False
        if genai and os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.ready = True

    def is_ready(self) -> bool:
        return self.ready

    def list_models(self) -> Iterable[str]:
        if not self.is_ready():
            return []
        try:
            for m in genai.list_models():
                yield (m.name if hasattr(m, "name") else str(m)).lower()
        except Exception:
            return []

    def run(self, prompt: str, model_id: str) -> Tuple[str, float]:
        if not self.is_ready():
            raise RuntimeError("Gemini not initialized")
        mid = model_id.split("/")[-1] if "/" in model_id else model_id
        m = genai.GenerativeModel(mid)
        t0 = time.time()
        out = m.generate_content(prompt)
        txt = getattr(out, "text", "") or ""
        if not txt and getattr(out, "candidates", None):
            chunks = []
            for c in out.candidates:
                if getattr(
                    c, "content", None
                    ) and getattr(
                        c.content, "parts", None
                        ):
                    for p in c.content.parts:
                        if hasattr(p, "text"):
                            chunks.append(p.text)
            txt = "\n".join(chunks)
        return txt, time.time() - t0
