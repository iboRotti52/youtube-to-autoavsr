from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ProcessingProfile:
    name: str
    require_talknet: bool

PROFILES = {
    "no_voiceover": ProcessingProfile(
        name="no_voiceover",
        require_talknet=False,
    ),
    "voiceover": ProcessingProfile(
        name="voiceover",
        require_talknet=True,
    ),
}

def get_profile(name: str) -> ProcessingProfile:
    if name not in PROFILES:
        raise ValueError(f"Unknown profile: {name}")
    return PROFILES[name]
