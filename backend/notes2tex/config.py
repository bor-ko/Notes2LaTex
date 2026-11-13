"""Paths and settings. Everything can be overridden with environment variables or .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.environ.get("NOTES2TEX_DATA", ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("NOTES2TEX_OUTPUT", ROOT / "output"))
SUBJECTS_DIR = Path(os.environ.get("NOTES2TEX_SUBJECTS", ROOT / "subjects"))
DB_PATH = Path(os.environ.get("NOTES2TEX_DB", DATA_DIR / "notes.db"))
FRONTEND_DIST = ROOT / "frontend" / "dist"

# Which engine transcribes by default: ollama (free, local) | claude | offline
ENGINE = os.environ.get("NOTES2TEX_ENGINE", "ollama")

# Local engine (Ollama)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST
LOCAL_MODEL = os.environ.get("NOTES2TEX_LOCAL_MODEL", "qwen3-vl:8b-instruct")
LOCAL_MODE = os.environ.get("NOTES2TEX_LOCAL_MODE", "line")  # line | page

# Claude engine (paid API)
MODEL = os.environ.get("NOTES2TEX_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("NOTES2TEX_EFFORT", "high")

# Rasterisation resolution for PDF exports. 200 DPI keeps pen strokes crisp
# without making page images huge.
RENDER_DPI = int(os.environ.get("NOTES2TEX_DPI", "200"))

# Long edge of the page image sent to Claude. Larger pages are downscaled.
MAX_IMAGE_EDGE = int(os.environ.get("NOTES2TEX_MAX_IMAGE_EDGE", "2000"))
