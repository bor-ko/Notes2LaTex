"""Compile .tex to PDF with tectonic and parse its errors."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_ERROR = re.compile(r"^error: (?P<file>[^:]+\.tex):(?P<line>\d+): (?P<msg>.+)$")


@dataclass
class TexError:
    line: int | None
    message: str


@dataclass
class CompileResult:
    ok: bool
    pdf_path: Path | None
    errors: list[TexError] = field(default_factory=list)
    log: str = ""


def tectonic_path() -> str:
    path = shutil.which("tectonic")
    if not path:
        raise RuntimeError("tectonic is not installed. Install it with: brew install tectonic")
    return path


def compile_tex(tex_path: Path, timeout: int = 240) -> CompileResult:
    tex_path = tex_path.resolve()
    proc = subprocess.run(
        [tectonic_path(), "-X", "compile", tex_path.name],
        cwd=tex_path.parent,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    log = proc.stdout + proc.stderr
    errors: list[TexError] = []
    for raw in log.splitlines():
        m = _ERROR.match(raw.strip())
        if m and m.group("file").endswith(tex_path.name):
            errors.append(TexError(int(m.group("line")), m.group("msg")))
        elif raw.startswith("error:") and "halted on" not in raw:
            errors.append(TexError(None, raw.removeprefix("error:").strip()))
    pdf = tex_path.with_suffix(".pdf")
    ok = proc.returncode == 0 and pdf.exists()
    if ok:
        errors = []
    elif not errors:
        errors.append(TexError(None, "tectonic failed; see log"))
    return CompileResult(ok=ok, pdf_path=pdf if pdf.exists() else None, errors=errors, log=log)
