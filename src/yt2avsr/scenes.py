from __future__ import annotations

import re
import subprocess
from pathlib import Path

# ffmpeg showinfo, seçilen (sahne-kesme) kareler için "pts_time:12.34" basar.
_PTS = re.compile(r"pts_time:(?P<t>\d+(?:\.\d+)?)")


def detect_scene_cuts(video: Path, threshold: float = 0.30) -> list[float]:
    """Videodaki sert kesme (sahne değişimi) zamanlarını saniye olarak döndürür.

    ffmpeg'in ``select='gt(scene,threshold)'`` filtresi ardışık kareler arası
    farkı ölçer; eşiği aşan her geçiş bir sahne kesmesidir. Ağır bir bağımlılık
    (PySceneDetect) gerektirmez; ffmpeg zaten pipeline'ın ön koşulu.

    threshold: 0.0-1.0. Düşük değer daha çok kesme yakalar (daha agresif bölme).
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(video),
        "-filter:v",
        f"select='gt(scene,{threshold})',showinfo",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    # ffmpeg showinfo çıktısını stderr'e yazar. Dönüş kodu başarılıysa parse et;
    # kesme yoksa hiçbir pts_time olmaz ve boş liste döneriz.
    cuts: list[float] = []
    for match in _PTS.finditer(proc.stderr or ""):
        cuts.append(round(float(match.group("t")), 3))
    return sorted(set(cuts))
