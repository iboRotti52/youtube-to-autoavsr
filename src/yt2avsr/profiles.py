from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ProcessingProfile:
    name: str
    # The ONLY difference between the two source lists:
    #   voiceover     -> external voice possible, so verify lip-sync and reject
    #                    segments where audio is active but the mouth is not moving.
    #   no_voiceover  -> audio matches the on-screen speaker, so the lip-sync
    #                    rejection is relaxed (natural pauses won't drop a segment).
    # Mouth/lip visibility, scene-cut and occlusion checks run identically in both.
    verify_lip_sync: bool

PROFILES = {
    "no_voiceover": ProcessingProfile(
        name="no_voiceover",
        verify_lip_sync=False,
    ),
    "voiceover": ProcessingProfile(
        name="voiceover",
        verify_lip_sync=True,
    ),
}

def get_profile(name: str) -> ProcessingProfile:
    if name not in PROFILES:
        raise ValueError(f"Unknown profile: {name}")
    return PROFILES[name]
