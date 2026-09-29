"""Configuration loading. `config.toml` is the single source of truth for bins and paths."""
from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    data: dict
    root: Path
    text_sha256: str

    def __getitem__(self, key):
        return self.data[key]

    def path(self, name: str) -> Path:
        p = self.root / self.data["paths"][name]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def core(self) -> list[str]:
        return list(self.data["panels"]["core"])

    @property
    def morphology(self) -> list[str]:
        return list(self.data["panels"]["morphology"])


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else ROOT / "config.toml"
    raw = p.read_bytes()
    return Config(data=tomllib.loads(raw.decode("utf8")), root=p.parent, text_sha256=hashlib.sha256(raw).hexdigest())
