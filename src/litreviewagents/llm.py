"""
OpenAI-compatible LLM client.

Works with any endpoint speaking the OpenAI /v1/chat/completions protocol:
- Ollama         (http://localhost:11434/v1)
- vLLM           (http://localhost:8000/v1)
- LM Studio      (http://localhost:1234/v1)
- OpenAI         (https://api.openai.com/v1)
- Together       (https://api.together.xyz/v1)
- Groq           (https://api.groq.com/openai/v1)
- Anthropic via litellm-proxy or similar

Local reasoning-heavy models (qwen3, deepseek-r1) emit chain-of-thought even
when `think:false` is set. `strip_reasoning_preamble` cleans the output.
"""

import json
import re
import ssl
import time
import urllib.request
from typing import Optional

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


# ── Reasoning preamble cleanup ────────────────────────────────────────────────

_PREAMBLE_PHRASES = [
    "okay, let's", "okay, i need", "okay, the user", "okay, so",
    "let me think", "let me break", "let me tackle", "let me check",
    "first, i need", "first, let's", "first, looking",
    "alright, let's", "alright, i need",
    "so, the user", "so, let's",
    "the user wants me to",
    "i need to figure out",
    "let's tackle this",
    "now, putting this together",
    "wait, the user",
    "hmm, maybe",
]

_CONTENT_MARKER = re.compile(
    r"(?:^|\n)"
    r"(?:"
    r"#{1,3} "
    r"|\*\*[A-Z1-9\[]"
    r"|RQ\d"
    r"|PAPER \["
    r"|SECTION:"
    r"|\d+\. \*\*"
    r"|\[\d+\]"
    r")",
    re.MULTILINE,
)


def _detect_infinite_loop(text: str) -> bool:
    """Detect when an 8-word phrase repeats 4+ times."""
    words = text.lower().split()
    if len(words) < 20:
        return False
    seen = {}
    window = 8
    for i in range(len(words) - window):
        phrase = " ".join(words[i:i + window])
        seen[phrase] = seen.get(phrase, 0) + 1
        if seen[phrase] >= 4:
            return True
    return False


def strip_reasoning_preamble(text: str) -> str:
    """
    Clean reasoning-model output:
    - Strips preambles like "Okay, let's tackle this..."
    - Detects and salvages from infinite reasoning loops
    - Truncates trailing incomplete deliberation
    """
    if not text:
        return text

    lower = text.lower().strip()
    starts_with_preamble = any(lower.startswith(p) for p in _PREAMBLE_PHRASES)
    loop_indicators = ["wait, ", "hmm, ", "but the user", "alternatively, maybe"]
    has_loops = sum(text.lower().count(ind) for ind in loop_indicators) >= 3
    has_infinite = _detect_infinite_loop(text)

    if not starts_with_preamble and not has_loops:
        return text

    # Infinite loop — try to salvage a paragraph
    if has_infinite:
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
        for para in paragraphs:
            if not any(p in para.lower() for p in _PREAMBLE_PHRASES[:8]):
                return (para + "\n\n[Note: Truncated — model entered a reasoning loop. "
                               "Consider refining the query.]")
        return ("[Analysis unavailable — model entered an infinite reasoning loop. "
                "Try refining the query or using a different model.]")

    # Find first content marker and return from there
    earliest = None
    for match in _CONTENT_MARKER.finditer(text):
        if match.start() > 20:
            earliest = match.start()
            break

    if earliest is not None:
        stripped = text[earliest:].strip()
        last_complete = max(
            stripped.rfind(".\n"), stripped.rfind("!\n"),
            stripped.rfind("?\n"), stripped.rfind("---"),
            stripped.rfind("**\n"),
        )
        if last_complete > len(stripped) * 0.5:
            stripped = stripped[:last_complete + 1].strip()
        return stripped if stripped else text

    return text[:400] + "... [truncated — no structured content found]"


# ── LLM Client ────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Minimal OpenAI-compatible chat client.

    Parameters
    ----------
    base_url : str
        e.g. "http://localhost:11434/v1" (Ollama)
             "https://api.openai.com/v1"
    model : str
        Model identifier, e.g. "qwen3:14b", "gpt-4o-mini", "llama-3.1-70b"
    api_key : str, optional
        Bearer token. Defaults to "ollama" if not provided (local endpoints).
    timeout : int
        Request timeout in seconds.
    max_retries : int
        Number of retries on failure.
    retry_wait : int
        Seconds between retries.
    temperature : float
        Sampling temperature.
    extra_options : dict, optional
        Backend-specific options passed in the "options" field
        (e.g. {"think": False} for Ollama qwen3).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "ollama",
        timeout: int = 600,
        max_retries: int = 1,
        retry_wait: int = 20,
        temperature: float = 0.2,
        extra_options: Optional[dict] = None,
        strip_preamble: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.temperature = temperature
        self.extra_options = extra_options or {}
        self.strip_preamble = strip_preamble

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
        """Send a chat request and return the assistant's content."""
        return self._call(system_prompt, user_prompt, max_tokens, attempt=0)

    def _call(self, system_prompt, user_prompt, max_tokens, attempt):
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        if self.extra_options:
            body["options"] = {**self.extra_options, "num_predict": max_tokens}

        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            print(f"    [LLM] attempt {attempt + 1}/{self.max_retries + 1} "
                  f"(timeout={self.timeout}s, max_tokens={max_tokens})")
            with urllib.request.urlopen(req, timeout=self.timeout, context=_SSL_CTX) as resp:
                raw = resp.read()
                data = json.loads(raw)
                choices = data.get("choices", [])
                if not choices:
                    raise ValueError(f"Empty choices: {raw[:200]}")
                msg = choices[0].get("message", {})
                content = msg.get("content", "").strip()
                if not content:
                    reasoning = msg.get("reasoning", "").strip()
                    if reasoning:
                        content = reasoning
                    else:
                        raise ValueError(f"Empty response: {raw[:200]}")
                if self.strip_preamble:
                    content = strip_reasoning_preamble(content)
                if not content:
                    raise ValueError("Content empty after preamble stripping")
                tokens = data.get("usage", {}).get("completion_tokens", "?")
                finish = choices[0].get("finish_reason", "?")
                print(f"    [LLM] ok — {tokens} tokens | finish: {finish}")
                return content
        except Exception as e:
            print(f"    [LLM] attempt {attempt + 1} failed: {e.__class__.__name__}: {e}")
            if attempt < self.max_retries:
                print(f"    [LLM] retrying in {self.retry_wait}s...")
                time.sleep(self.retry_wait)
                return self._call(system_prompt, user_prompt, max_tokens, attempt + 1)
            return f"[LLM error after {attempt + 1} attempt(s): {e.__class__.__name__}: {e}]"


# ── Module-level convenience ───────────────────────────────────────────────────

_default_client: Optional[LLMClient] = None


def configure(base_url, model, **kwargs):
    """Set the module-level default client."""
    global _default_client
    _default_client = LLMClient(base_url=base_url, model=model, **kwargs)
    return _default_client


def llm_call(system_prompt, user_prompt, max_tokens=500):
    """Convenience function using the module-level client."""
    if _default_client is None:
        raise RuntimeError("Call llm.configure(...) first, or use LLMClient directly.")
    return _default_client.chat(system_prompt, user_prompt, max_tokens)
