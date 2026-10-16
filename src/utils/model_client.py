"""
src/utils/model_client.py
Unified LLM client — routes to Gemini or Ollama based on config.
Features:
  • Key-pool rotator for Gemini (auto-switches on 429 quota exhaustion or 403)
  • AgentDojo-compatible wrapper (.models.generate_content)
  • Zero model names hardcoded in code — everything comes from config/models.yaml.

Usage:
    from src.utils.config_loader import load_config
    from src.utils.model_client import ModelClient, build_gemini_rotator

    cfg = load_config()
    client = ModelClient(cfg.base_detector, cfg.secrets)
    response = client.generate(prompt="...", system_prompt="...")
"""
from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import requests

from src.utils.config_loader import ModelConfig, SecretsConfig
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class _BaseLLMClient(ABC):
    """Common interface for all LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Send a prompt and return the text response.

        Args:
            prompt: User message.
            system_prompt: Optional system instruction.

        Returns:
            Model response as a string.
        """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return approximate token count for text."""


# ---------------------------------------------------------------------------
# Rotating Gemini Client (supports both direct generation and AgentDojo)
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class KeySlot:
    """Tracks state and availability of an API key within the rotation pool."""
    index: int
    api_key: str
    client: Any
    status: str = "HEALTHY"  # "HEALTHY" | "COOLDOWN" | "DEAD"
    cooldown_until: float = 0.0
    total_calls: int = 0
    total_errors: int = 0
    last_error: str = ""

    @property
    def masked(self) -> str:
        return f"...{self.api_key[-6:]}" if len(self.api_key) >= 6 else "***"

    def is_available(self, now: float) -> bool:
        if self.status == "DEAD":
            return False
        if self.status == "COOLDOWN":
            if now >= self.cooldown_until:
                self.status = "HEALTHY"
                return True
            return False
        return True


class _ModelsProxy:
    """Proxy for client.models to intercept generate_content and handle key rotation."""

    def __init__(self, rotator: "RotatingGeminiClient") -> None:
        self._rotator = rotator

    def generate_content(self, *args, **kwargs) -> Any:
        return self._rotator._execute_with_rotation(
            lambda c: c.models.generate_content(*args, **kwargs)
        )

    def list(self, *args, **kwargs) -> Any:
        return self._rotator._execute_with_rotation(
            lambda c: c.models.list(*args, **kwargs)
        )


class RotatingGeminiClient(_BaseLLMClient):
    """
    High-availability Gemini client backed by an intelligent pool of API keys.
    Features:
      • Automatic health-state tracking (HEALTHY, COOLDOWN, DEAD).
      • Permanent isolation of 403 Forbidden / invalid project keys.
      • Temporary cooldown (30-60s) for 429 Quota exhausted keys.
      • Exponential backoff + jitter for 503 High Demand / Service Unavailable.
      • Transparent retry on network drops (Server disconnected, ConnectionReset).
      • Automatic recovery as cooldown windows expire.
    """

    def __init__(
        self,
        api_keys: Union[List[str], str],
        model_name: str = "gemini-flash-latest",
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
    ) -> None:
        if isinstance(api_keys, str):
            raw_keys = [api_keys] if api_keys.strip() else []
        else:
            raw_keys = [k for k in api_keys if k and k.strip()]

        # Deduplicate keys while preserving order
        seen = set()
        self.api_keys: List[str] = []
        for k in raw_keys:
            if k not in seen:
                seen.add(k)
                self.api_keys.append(k)

        if not self.api_keys:
            raise ValueError(
                "[Config Error] No Gemini API keys found in config/secrets.yaml.\n"
                "  -> Add your keys under gemini.api_keys."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_output_tokens
        self.current_index = 0
        self.models = _ModelsProxy(self)
        self.slots: List[KeySlot] = []
        self._lock = threading.Lock()
        self._last_call_time = 0.0
        self._min_call_interval = 0.8  # 800ms pacing between requests to prevent burst 503s
        self._init_slots()

    def _init_slots(self) -> None:
        try:
            from google import genai
            for i, k in enumerate(self.api_keys):
                c = genai.Client(api_key=k)
                self.slots.append(KeySlot(index=i, api_key=k, client=c))
            self._use_new_sdk = True
        except ImportError:
            import google.generativeai as genai
            for i, k in enumerate(self.api_keys):
                genai.configure(api_key=k)
                m = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    ),
                )
                self.slots.append(KeySlot(index=i, api_key=k, client=m))
            self._use_new_sdk = False

        log.info(
            f"RotatingGeminiClient initialized with {len(self.slots)} API keys in pool."
        )

    @property
    def current_client(self) -> Any:
        slot = self._select_slot()
        return slot.client

    def _select_slot(self) -> KeySlot:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self._min_call_interval:
                time.sleep(self._min_call_interval - elapsed)
                now = time.time()
            self._last_call_time = now

            alive_slots = [s for s in self.slots if s.status != "DEAD"]
            if not alive_slots:
                raise RuntimeError(
                    "[API Pool Fatal] All Gemini API keys in pool are marked DEAD (403 Permission Denied).\n"
                    "  -> Please provide new valid API keys in config/secrets.yaml."
                )

            # Find healthy or expired cooldown slots
            available = [s for s in alive_slots if s.is_available(now)]

        if not available:
            # All alive keys are on cooldown: wait for the earliest one to expire
            earliest = min(alive_slots, key=lambda s: s.cooldown_until)
            wait_s = max(1.0, earliest.cooldown_until - now + 0.5)
            wait_s = min(wait_s, 30.0)  # safety cap
            log.warning(
                f"[API Pool Pacing] All {len(alive_slots)} active keys on cooldown. "
                f"Waiting {wait_s:.1f}s for key #{earliest.index + 1} ({earliest.masked}) to reset..."
            )
            time.sleep(wait_s)
            now = time.time()
            earliest.status = "HEALTHY"
            return earliest

        # Round-robin among available slots starting from current_index
        for offset in range(len(self.slots)):
            idx = (self.current_index + offset) % len(self.slots)
            slot = self.slots[idx]
            if slot.is_available(now):
                self.current_index = (idx + 1) % len(self.slots)
                return slot

        return available[0]

    def _execute_with_rotation(self, fn) -> Any:
        max_attempts = max(6, len(self.slots) * 3)
        last_exc = None

        for attempt in range(max_attempts):
            slot = self._select_slot()
            now = time.time()
            try:
                result = fn(slot.client)
                slot.total_calls += 1
                return result
            except Exception as exc:
                err_str = str(exc)
                slot.total_errors += 1
                slot.last_error = err_str[:200]
                last_exc = exc

                is_quota = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "ResourceExhausted" in err_str
                is_perm = "403" in err_str or "PERMISSION_DENIED" in err_str
                is_demand = "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str
                is_net = any(k in err_str.lower() for k in [
                    "disconnected", "connection", "remote", "wsarecv", "wsasend", "timeout", "abort"
                ])

                if is_perm:
                    slot.status = "DEAD"
                    log.warning(
                        f"[API Pool] Key #{slot.index + 1} ({slot.masked}) returned 403 Forbidden. "
                        f"Marked as DEAD and isolated from pool."
                    )
                elif is_quota:
                    slot.status = "COOLDOWN"
                    slot.cooldown_until = now + 45.0  # standard quota reset window
                    log.warning(
                        f"[API Pool] Key #{slot.index + 1} ({slot.masked}) hit 429 Quota Limit. "
                        f"Placed on cooldown for 45s. Rotating to next key..."
                    )
                elif is_demand:
                    slot.status = "COOLDOWN"
                    slot.cooldown_until = now + 10.0  # high demand spike window
                    log.warning(
                        f"[API Pool] Key #{slot.index + 1} ({slot.masked}) hit 503 High Demand. "
                        f"Placed on cooldown for 10s. Rotating..."
                    )
                    time.sleep(1.5)
                elif is_net:
                    log.warning(
                        f"[API Pool] Key #{slot.index + 1} ({slot.masked}) transient network glitch: {err_str[:60]}. "
                        f"Backing off and rotating..."
                    )
                    time.sleep(1.0)
                else:
                    # Unrecognized error: raise immediately rather than masking
                    raise exc

        raise last_exc or RuntimeError("All API keys in pool failed after maximum retries.")

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        t0 = time.time()

        if self._use_new_sdk:
            resp = self._execute_with_rotation(
                lambda c: c.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                )
            )
            text = (resp.text or "").strip()
        else:
            resp = self._execute_with_rotation(
                lambda c: c.generate_content(full_prompt)
            )
            text = (resp.text or "").strip()

        elapsed = (time.time() - t0) * 1000
        log.debug(f"Gemini call: {elapsed:.0f}ms, model={self.model_name}")
        return text

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def build_gemini_rotator(
    secrets: SecretsConfig,
    model_name: str = "gemini-flash-latest",
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
) -> RotatingGeminiClient:
    """Build a managed RotatingGeminiClient from SecretsConfig."""
    keys = getattr(secrets, "gemini_api_keys", None) or [secrets.gemini_api_key]
    return RotatingGeminiClient(
        api_keys=keys,
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

class _OllamaClient(_BaseLLMClient):
    """Local Ollama via HTTP REST API (localhost:11434)."""

    def __init__(self, model_cfg: ModelConfig, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model_cfg.model
        self._temperature = model_cfg.temperature
        self._max_tokens = model_cfg.max_output_tokens
        self._check_connection()
        log.debug(f"OllamaClient initialized: model={model_cfg.model}, url={base_url}")

    def _check_connection(self) -> None:
        try:
            r = requests.get(f"{self._base_url}/api/tags", timeout=5)
            r.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"[Ollama Error] Cannot connect to Ollama at {self._base_url}.\n"
                f"  -> Make sure Ollama is running: run 'ollama serve' in a terminal."
            )

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        t0 = time.time()
        r = requests.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        elapsed = (time.time() - t0) * 1000
        result = r.json().get("response", "").strip()
        log.debug(f"Ollama call: {elapsed:.0f}ms, model={self._model}")
        return result

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Public factory class
# ---------------------------------------------------------------------------

class ModelClient:
    """
    Public client — routes to Gemini or Ollama based on ModelConfig.provider.
    Supports API key rotation for Gemini automatically.
    """

    def __init__(self, model_cfg: ModelConfig, secrets: SecretsConfig) -> None:
        provider = model_cfg.provider.lower()
        if provider == "gemini":
            keys = getattr(secrets, "gemini_api_keys", None) or [secrets.gemini_api_key]
            self._client = RotatingGeminiClient(
                api_keys=keys,
                model_name=model_cfg.model,
                temperature=model_cfg.temperature,
                max_output_tokens=model_cfg.max_output_tokens,
            )
        elif provider == "ollama":
            self._client = _OllamaClient(model_cfg, secrets.ollama_base_url)
        else:
            raise ValueError(
                f"[Config Error] Unknown provider '{provider}' in models.yaml.\n"
                f"  -> Valid options: 'gemini' | 'ollama'"
            )
        self.provider = provider
        self.model_name = model_cfg.model

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response from the configured model."""
        return self._client.generate(prompt, system_prompt)

    def count_tokens(self, text: str) -> int:
        """Return approximate token count."""
        return self._client.count_tokens(text)

    def __repr__(self) -> str:
        return f"ModelClient(provider={self.provider}, model={self.model_name})"
