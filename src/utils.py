import re
from typing import Iterable, Optional


def normalize_model_name(name: str) -> str:
    n = name.lower()
    n = n.replace("'", "")
    n = re.sub(r"[\-_/.:]", " ", n)
    n = re.sub(r"models?", "", n)
    n = re.sub(r"meta|together|openai|bedrock|gemini|anthropic", "", n)
    n = re.sub(r"llama", " llama ", n)
    n = re.sub(r"instruct", " instruct ", n)
    n = re.sub(r"turbo", " turbo ", n)
    n = re.sub(r"pro", " pro ", n)
    n = re.sub(r"flash", " flash ", n)
    n = re.sub(r"haiku", " haiku ", n)
    n = re.sub(r"sonnet", " sonnet ", n)
    n = re.sub(r"chatgpt", " chatgpt ", n)
    n = re.sub(r"claude", " claude ", n)
    n = re.sub(r"black forest labs", "black-forest-labs", n)
    n = re.sub(r"\s+", " ", n).strip()
    n = n.replace("3 5", "3.5").replace("3 1", "3.1").replace("3 3", "3.3")
    n = n.replace("8b", "8b").replace("70b", "70b")
    return n


def fuzzy_match_model(
    friendly_name: str, provider_models: Iterable[str]
) -> Optional[str]:
    norm_friendly = normalize_model_name(friendly_name)
    friendly_tokens = set(norm_friendly.split())
    best_score = 0
    best_model = None
    for model_id in provider_models:
        norm_model = normalize_model_name(model_id)
        model_tokens = set(norm_model.split())
        if norm_friendly == norm_model:
            return model_id
        score = len(friendly_tokens & model_tokens)
        if score > best_score:
            best_score, best_model = score, model_id
    if best_score >= max(2, len(friendly_tokens) // 2):
        return best_model
    for model_id in provider_models:
        if (
            norm_friendly in normalize_model_name(model_id)
            or normalize_model_name(model_id) in norm_friendly
        ):
            return model_id
    return None
