#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = [
    ("datasets_pc_parts", "datasets_pc"),
    ("datasets_pc_sft_parts", "datasets_pc_sft"),
]

for parts_dir_name, out_dir_name in SPECS:
    parts_dir = ROOT / parts_dir_name
    out_dir = ROOT / out_dir_name
    if not parts_dir.exists():
        continue
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted({p.name.split(".part-")[0] for p in parts_dir.glob("*.part-*")})
    for stem in stems:
        out_path = out_dir / stem
        with out_path.open("wb") as out_f:
            for part in sorted(parts_dir.glob(f"{stem}.part-*")):
                out_f.write(part.read_bytes())
        print(f"wrote {out_path.relative_to(ROOT)}")
