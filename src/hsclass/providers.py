"""
Classification providers. A provider takes a batch of (key, description) pairs and returns, for the
items it could answer, (hs_chapter, confidence). Items it cannot answer are simply left out, and the
pipeline hands them to the next provider.

* AnthropicProvider  - Claude via the official SDK (ANTHROPIC_API_KEY, optional CLAUDE_MODEL)
* GroqProvider       - Groq's OpenAI-compatible chat API (GROQ_API_KEY, optional GROQ_MODEL)
* HeuristicProvider  - offline keyword matcher so the demo runs with no keys. It is NOT an LLM and
                       its accuracy says nothing about LLM accuracy.
"""
from __future__ import annotations

import json
import os
import re
import time

from .hs_reference import VALID_CHAPTERS, chapter_prompt_table
from .normalise import tokens

Answer = tuple[int, float]  # (hs_chapter, confidence 0..1)

DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"  # override with GROQ_MODEL if Groq retires it


class ProviderError(RuntimeError):
    """The provider failed for the whole batch (network, auth, unparseable reply)."""


# ---------------------------------------------------------------------------- prompt + parsing

SYSTEM_PROMPT = (
    "You are a customs classification assistant. For each cargo description, choose the Harmonized "
    "System CHAPTER (a number from 1 to 97, never 77) that best fits the goods. Descriptions are "
    "messy free text from shipping documents: abbreviations, typos, quantities and container codes. "
    "Judge only the goods described. If a description is too vague or could fit several chapters, "
    "return a LOW confidence instead of guessing. Reply with JSON only."
)


def build_user_prompt(batch: list[tuple[str, str]]) -> tuple[str, dict[str, str]]:
    """Returns (prompt, id_to_key). Short numeric ids keep the reply small and easy to validate."""
    id_to_key = {str(i): key for i, (key, _) in enumerate(batch, start=1)}
    items = [{"id": str(i), "description": text} for i, (_, text) in enumerate(batch, start=1)]
    prompt = (
        "HS chapters:\n" + chapter_prompt_table() + "\n\n"
        "Classify these cargo descriptions:\n" + json.dumps(items, ensure_ascii=True) + "\n\n"
        'Return a JSON object exactly like {"results": [{"id": "1", "chapter": 2, "confidence": 0.93}]} '
        "with one entry per id. confidence is between 0 and 1."
    )
    return prompt, id_to_key


def parse_llm_response(text: str, id_to_key: dict[str, str]) -> dict[str, Answer]:
    """Validate a model reply. Bad or unknown entries are dropped, never trusted."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"reply was not valid JSON: {exc}") from exc
    results = data.get("results") if isinstance(data, dict) else data
    if not isinstance(results, list):
        raise ProviderError("reply had no 'results' list")

    out: dict[str, Answer] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        key = id_to_key.get(str(item.get("id")))
        if key is None or key in out:
            continue
        try:
            chapter = int(item.get("chapter"))
            conf = float(item.get("confidence"))
        except (TypeError, ValueError):
            continue
        if chapter not in VALID_CHAPTERS or conf != conf:  # also rejects NaN
            continue
        out[key] = (chapter, min(max(conf, 0.0), 1.0))
    return out


# ---------------------------------------------------------------------------- LLM providers

class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def classify(self, batch: list[tuple[str, str]]) -> dict[str, Answer]:
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError("the 'anthropic' package is not installed") from exc
        prompt, id_to_key = build_user_prompt(batch)
        try:
            client = anthropic.Anthropic(max_retries=3)
            msg = client.messages.create(
                model=self.model, max_tokens=2000, temperature=0, system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # SDK raises several error types; all mean "this batch failed"
            raise ProviderError(f"anthropic request failed: {type(exc).__name__}: {exc}") from exc
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return parse_llm_response(text, id_to_key)


class GroqProvider:
    name = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("GROQ_API_KEY"))

    def classify(self, batch: list[tuple[str, str]]) -> dict[str, Answer]:
        import requests

        prompt, id_to_key = build_user_prompt(batch)
        payload = {
            "model": self.model, "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}],
        }
        headers = {"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY', '')}"}
        last = "no attempt made"
        for attempt in range(1, 4):
            try:
                resp = requests.post(self.URL, json=payload, headers=headers, timeout=60)
                if resp.status_code in (429, 500, 502, 503, 504):
                    last = f"HTTP {resp.status_code}"
                    time.sleep(2 * attempt)
                    continue
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
                return parse_llm_response(text, id_to_key)
            except ProviderError:
                raise
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
                time.sleep(2 * attempt)
        raise ProviderError(f"groq request failed after 3 attempts ({last})")


# ---------------------------------------------------------------------------- offline stand-in

# Deliberately small and imperfect. Single words are matched on singularised tokens; multi-word
# entries are matched as phrases. Chapters with no entry here (for example 29) always go to review.
_KEYWORDS: dict[int, list[str]] = {
    2: ["beef", "mutton", "chicken", "poultry", "buffalo meat", "pork"],
    3: ["shrimp", "salmon", "tuna", "fish", "prawn", "squid"],
    4: ["milk", "butter", "cheese", "honey", "egg", "yoghurt"],
    7: ["onion", "potato", "green pea", "chickpea", "lentil", "carrot"],
    8: ["banana", "orange", "date", "cashew", "almond", "apple", "grape", "walnut", "mango"],
    9: ["coffee", "tea", "pepper", "cardamom", "turmeric", "cumin", "clove"],
    10: ["wheat", "rice", "basmati", "maize", "barley", "sorghum", "corn"],
    11: ["flour", "starch", "malt", "wheat flour"],
    12: ["soybean", "sesame", "sunflower seed", "groundnut", "peanut"],
    15: ["palm oil", "olive oil", "soybean oil", "sunflower oil", "cooking oil", "coconut oil"],
    16: ["canned tuna", "canned sardine", "sausage"],
    17: ["sugar", "confectionery", "sweet", "candy", "icumsa"],
    18: ["cocoa", "chocolate"],
    19: ["pasta", "spaghetti", "biscuit", "breakfast cereal", "noodle"],
    20: ["tomato paste", "fruit juice", "canned pineapple", "pickle", "juice concentrate", "jam"],
    21: ["instant coffee", "soup", "soy sauce", "ice cream", "ketchup"],
    22: ["mineral water", "soft drink", "energy drink", "wine", "vinegar", "beer"],
    23: ["soybean meal", "bran", "ddg", "dog food", "animal feed", "fish meal"],
    24: ["tobacco", "cigarette"],
    25: ["cement", "gypsum", "silica sand", "salt", "granite"],
    26: ["iron ore", "copper concentrate", "bauxite"],
    27: ["coal", "lpg", "bitumen", "diesel", "gasoil", "petroleum coke", "kerosene", "naphtha"],
    28: ["caustic soda", "sulphuric acid", "soda ash"],
    30: ["pharmaceutical", "medicament", "vaccine"],
    31: ["urea", "npk", "dap", "fertilizer", "fertiliser"],
    32: ["paint", "pigment", "printing ink"],
    33: ["perfume", "shampoo", "cosmetic", "skin cream"],
    34: ["detergent", "soap", "candle"],
    38: ["disinfectant", "antifreeze", "pesticide"],
    39: ["polyethylene", "pvc", "polypropylene", "plastic"],
    40: ["tyre", "tire", "rubber glove", "natural rubber", "rubber"],
    41: ["leather"],
    42: ["handbag", "suitcase"],
    44: ["timber", "plywood", "wooden pallet", "wood pellet", "lumber", "sawn"],
    47: ["wood pulp", "waste paper"],
    48: ["paper", "kraft", "corrugated", "tissue"],
    49: ["book", "brochure", "newspaper"],
    52: ["cotton"],
    54: ["filament"],
    55: ["staple"],
    57: ["carpet", "rug"],
    61: ["knitted", "sweater", "shirt"],
    62: ["trouser", "dress", "jacket"],
    63: ["bed linen", "towel", "used clothing", "worn clothing"],
    64: ["shoe", "footwear", "sneaker"],
    68: ["marble"],
    69: ["ceramic", "porcelain", "sanitary ware"],
    70: ["glass", "glassware", "float glass"],
    71: ["jewellery", "jewelry"],
    72: ["steel", "hot rolled", "rebar", "stainless steel", "pig iron", "steel coil"],
    73: ["steel pipe", "wire rope", "fastener", "bolt", "fabricated"],
    74: ["copper"],
    76: ["aluminium", "aluminum"],
    79: ["zinc"],
    82: ["spanner", "cutlery", "hand tool"],
    83: ["padlock"],
    84: ["pump", "air conditioner", "excavator", "crane", "laptop", "refrigerator", "compressor", "boiler"],
    85: ["mobile phone", "smartphone", "electric motor", "television", "lithium batter", "battery",
         "solar panel", "generating set", "cable"],
    87: ["car", "truck", "motorcycle", "tractor", "brake pad"],
    88: ["aircraft"],
    89: ["tug boat", "yacht", "inflatable boat"],
    90: ["syringe", "surgical instrument", "measuring instrument"],
    91: ["watch"],
    94: ["furniture", "mattress", "lighting fixture", "prefab cabin"],
    95: ["toy", "sports equipment", "football"],
    96: ["pen", "brush", "zipper"],
    97: ["painting", "artwork"],
}


_ES_EXCEPTIONS = {"SHOES": "SHOE"}  # plain "-OES" stripping would give "SHO"


def _singular(tok: str) -> str:
    """Very small plural stripper: enough for commodity words, not a real stemmer."""
    if tok in _ES_EXCEPTIONS:
        return _ES_EXCEPTIONS[tok]
    if len(tok) > 4 and tok.endswith(("OES", "CHES", "SHES", "XES", "SSES")):
        return tok[:-2]  # POTATOES -> POTATO, WATCHES -> WATCH, MATTRESSES -> MATTRESS
    if len(tok) > 3 and tok.endswith("S") and not tok.endswith("SS"):
        return tok[:-1]
    return tok


def _stem_phrase(phrase: str) -> str:
    return " ".join(_singular(t) for t in phrase.upper().split())


_STEMMED: list[tuple[str, int]] = [(_stem_phrase(kw), ch) for ch, kws in _KEYWORDS.items() for kw in kws]


class HeuristicProvider:
    name = "heuristic"

    @staticmethod
    def available() -> bool:
        return True

    def classify(self, batch: list[tuple[str, str]]) -> dict[str, Answer]:
        out: dict[str, Answer] = {}
        for key, text in batch:
            toks = [_singular(t) for t in tokens(text)]
            joined = " " + " ".join(toks) + " "
            score: dict[int, int] = {}
            for phrase, chapter in _STEMMED:
                if f" {phrase} " in joined:
                    score[chapter] = score.get(chapter, 0) + len(phrase)
            if not score:
                continue  # unknown goods: leave for review
            ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
            top_ch, top = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else 0
            if second == 0:
                conf = 0.90
            elif top >= 2 * second:
                conf = 0.80
            else:
                conf = 0.55  # genuinely ambiguous: below the default threshold, so it goes to review
            out[key] = (top_ch, conf)
        return out


# ---------------------------------------------------------------------------- factory

_REGISTRY = {"anthropic": AnthropicProvider, "groq": GroqProvider, "heuristic": HeuristicProvider}


def build_providers(spec: str) -> list:
    """'auto' = Claude then Groq when their keys are set, else the offline heuristic.
    Otherwise a comma list such as 'anthropic,groq' or 'heuristic'."""
    if spec.strip().lower() == "auto":
        chosen = [n for n, cls in (("anthropic", AnthropicProvider), ("groq", GroqProvider)) if cls.available()]
        return [_REGISTRY[n]() for n in (chosen or ["heuristic"])]
    providers = []
    for name in (s.strip().lower() for s in spec.split(",") if s.strip()):
        if name not in _REGISTRY:
            raise ValueError(f"unknown provider '{name}' (choose from {', '.join(_REGISTRY)})")
        providers.append(_REGISTRY[name]())
    if not providers:
        raise ValueError("no providers selected")
    return providers
