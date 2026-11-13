"""Engine registry: pick a transcriber by name."""

from __future__ import annotations

from .. import config
from .base import OfflineTranscriber, Transcriber, TranscriptionError

ENGINES = {
    "ollama": "Local model (free)",
    "claude": "Claude API (paid)",
    "offline": "Offline (type it yourself)",
}


def get_transcriber(engine: str | None = None) -> Transcriber:
    engine = engine or config.ENGINE
    if engine == "offline":
        return OfflineTranscriber()
    if engine == "ollama":
        from .ollama import OllamaTranscriber

        return OllamaTranscriber()
    if engine == "claude":
        from .claude import ClaudeTranscriber

        return ClaudeTranscriber()
    raise TranscriptionError(f"Unknown engine '{engine}'. Choose one of: {', '.join(ENGINES)}")


def engine_status() -> list[dict]:
    """Which engines can run right now, and why not."""
    from .claude import has_credentials
    from .ollama import OllamaTranscriber

    out = []
    try:
        OllamaTranscriber(check=True)
        ollama = (True, config.LOCAL_MODEL)
    except TranscriptionError as e:
        ollama = (False, str(e))
    claude_ok = has_credentials()
    states = {
        "ollama": ollama,
        "claude": (claude_ok, config.MODEL if claude_ok else "No ANTHROPIC_API_KEY in .env"),
        "offline": (True, "placeholders"),
    }
    for name, label in ENGINES.items():
        ok, detail = states[name]
        out.append({"name": name, "label": label, "available": ok, "detail": detail})
    return out
