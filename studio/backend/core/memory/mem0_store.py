# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Privacy-first Mem0 adapter for Unsloth Studio.

Mem0 is optional at import time and never receives a conversation by accident.
The adapter uses an account-scoped local Qdrant directory and a local
HuggingFace embedder. The Mem0 LLM is pointed at Studio's local OpenAI-compatible
endpoint by default, and telemetry is disabled before importing Mem0. If the
optional package or local embedding runtime is absent, the existing bounded JSON
learning ledger remains the source of truth.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any

from utils.paths import account_path, ensure_dir

_LOCK = threading.RLock()
_INSTANCES: dict[str, Any] = {}
_MAX_EXPERIENCE_CHARS = 8_000
_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _root() -> Path:
    path = account_path("learning/mem0")
    ensure_dir(path)
    return path


def _user_id(subject: str | None) -> str:
    # Do not put an email or account subject into the vector store.
    digest = hashlib.sha256(str(subject or "local").encode("utf-8")).hexdigest()[:32]
    return f"unsloth-{digest}"


def _import_mem0() -> Any:
    # Mem0's default telemetry is on. Set this before any mem0 import, including
    # its package-level initialization.
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from mem0 import Memory

    return Memory


def _config() -> dict[str, Any]:
    root = _root()
    embedding_model = os.environ.get("UNSLOTH_MEM0_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL).strip()
    embedding_dims = int(os.environ.get("UNSLOTH_MEM0_EMBEDDING_DIMS", "384"))
    llm_model = os.environ.get("UNSLOTH_MEM0_LLM_MODEL", "local").strip() or "local"
    llm_base_url = os.environ.get("UNSLOTH_MEM0_LLM_BASE_URL", "http://127.0.0.1:8888/v1").strip()
    llm_provider = os.environ.get("UNSLOTH_MEM0_LLM_PROVIDER", "openai").strip().lower() or "openai"
    if llm_provider == "ollama":
        llm_config: dict[str, Any] = {
            "model": llm_model,
            "ollama_base_url": os.environ.get("UNSLOTH_MEM0_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        }
    else:
        llm_config = {
            "model": llm_model,
            # The placeholder key is never sent to a remote endpoint by this
            # adapter unless the user explicitly overrides the base URL.
            "api_key": os.environ.get("UNSLOTH_MEM0_LLM_API_KEY", "local"),
            "openai_base_url": llm_base_url,
        }
    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "unsloth_learning",
                "embedding_model_dims": embedding_dims,
                "path": str(root / "qdrant"),
            },
        },
        "llm": {"provider": llm_provider, "config": llm_config},
        "embedder": {
            "provider": "huggingface",
            "config": {"model": embedding_model, "embedding_dims": embedding_dims},
        },
        "history_db_path": str(root / "history.db"),
    }


def _instance() -> Any:
    key = str(_root()) + "\0" + os.environ.get("UNSLOTH_MEM0_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)
    with _LOCK:
        if key not in _INSTANCES:
            Memory = _import_mem0()
            _INSTANCES[key] = Memory.from_config(_config())
        return _INSTANCES[key]


def status() -> dict[str, Any]:
    try:
        import importlib.util

        installed = importlib.util.find_spec("mem0") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        installed = False
    configured = bool(os.environ.get("UNSLOTH_MEM0_LLM_BASE_URL", "http://127.0.0.1:8888/v1"))
    return {
        "installed": installed,
        "enabled": installed,
        "available": installed,
        "configured": configured,
        "backend": "mem0 + local qdrant + local HuggingFace embeddings" if installed else "bounded local learning ledger",
        "privacy": "Account-scoped local storage; Mem0 telemetry disabled; no remote endpoint unless explicitly configured.",
        "root": str(_root()),
        "error": None,
    }


def add_experience(subject: str | None, text: str, *, thread_id: str | None = None, kind: str = "experience") -> dict[str, Any]:
    text = str(text or "").strip()[:_MAX_EXPERIENCE_CHARS]
    if not text:
        return {"stored": False, "reason": "empty"}
    try:
        memory = _instance()
        result = memory.add(
            text,
            user_id=_user_id(subject),
            metadata={"source": "unsloth-studio", "kind": kind, "thread_id": str(thread_id or "")[:200]},
            infer=False,
        )
        return {"stored": True, "result": result}
    except Exception as error:  # noqa: BLE001 -- optional integration must never break chat
        return {"stored": False, "reason": str(error)[:1_000]}


def search(subject: str | None, query: str, limit: int = 5) -> dict[str, Any]:
    query = str(query or "").strip()[:2_000]
    if not query:
        return {"results": [], "available": False, "reason": "empty"}
    try:
        memory = _instance()
        results = memory.search(query, user_id=_user_id(subject), limit=max(1, min(int(limit), 10)))
        return {"results": results if isinstance(results, list) else [], "available": True}
    except Exception as error:  # noqa: BLE001 -- optional integration must never break chat
        return {"results": [], "available": False, "reason": str(error)[:1_000]}
