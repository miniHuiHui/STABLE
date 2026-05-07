#!/usr/bin/env python3
"""
Evaluate BrickGPT-style mean / min stability (ICCV 2025, Sec. 5 & Table 1).

For each *valid* generated structure, compute per-brick stability scores s_i
(higher = more stable; s_i = 0 unstable), then:

  - mean stability (per structure): mean_i s_i over bricks
  - min stability (per structure): min_i s_i (weakest brick)

Voxel heatmap scores come from `BrickStructure.stability_scores()` (Gurobi analysis).
We convert heatmap channel R (instability in [0,1]) to paper convention:

  s_voxel = clip(1 - R, 0, 1)

Per brick we take the bottleneck: s_brick = min_{voxels in brick} s_voxel.

Requires: Gurobi license (same as `brickgpt.stability_analysis.stability_score`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brickgpt.data.brick_structure import BrickStructure  # noqa: E402


def per_brick_paper_scores(structure: BrickStructure, voxel_scores: np.ndarray) -> np.ndarray:
    """One scalar s_i per brick (paper: higher is more stable)."""
    out: list[float] = []
    for brick in structure.bricks:
        patch = voxel_scores[brick.slice]
        s_vox = np.clip(1.0 - patch, 0.0, 1.0)
        out.append(float(np.min(s_vox)))
    return np.array(out, dtype=np.float64)


def analyze_structure(structure: BrickStructure) -> dict | None:
    if structure.has_collisions() or structure.has_floating_bricks() or structure.has_out_of_bounds_bricks():
        return None
    try:
        voxel_scores = structure.stability_scores()
    except Exception as e:
        return {"error": str(e)}
    # Solver failure returns an all-ones grid (see stability_analysis.stability_score).
    if np.allclose(voxel_scores, 1.0):
        return {"error": "stability_solve_failed_or_infeasible"}
    s_bricks = per_brick_paper_scores(structure, voxel_scores)
    if s_bricks.size == 0:
        return None
    return {
        "n_bricks": len(structure.bricks),
        "mean_stability": float(np.mean(s_bricks)),
        "min_stability": float(np.min(s_bricks)),
    }


def load_structure_from_ldr(path: Path) -> BrickStructure | None:
    try:
        text = path.read_text()
        if not text.strip():
            return None
        return BrickStructure.from_ldr(text)
    except Exception:
        return None


def eval_folder(exp_dir: Path) -> dict:
    ldr_files = sorted(exp_dir.glob("sample_*_gen.ldr"))
    rows: list[dict] = []
    mean_stabs: list[float] = []
    min_stabs: list[float] = []

    for ldr in ldr_files:
        st = load_structure_from_ldr(ldr)
        base = {"ldr": ldr.name, "parse_ok": st is not None}
        if st is None:
            rows.append({**base, "status": "parse_failed"})
            continue
        res = analyze_structure(st)
        if res is None:
            rows.append({**base, "status": "invalid_structure"})
            continue
        if "error" in res:
            rows.append({**base, "status": "stability_error", **res})
            continue
        mean_stabs.append(res["mean_stability"])
        min_stabs.append(res["min_stability"])
        rows.append(
            {
                **base,
                "status": "ok",
                "n_bricks": res["n_bricks"],
                "mean_stability": res["mean_stability"],
                "min_stability": res["min_stability"],
            }
        )

    summary = {
        "experiment_dir": str(exp_dir),
        "n_ldr_files": len(ldr_files),
        "n_ok": len(mean_stabs),
        "dataset_mean_stability": float(np.mean(mean_stabs)) if mean_stabs else None,
        "dataset_mean_min_stability": float(np.mean(min_stabs)) if min_stabs else None,
        "std_mean_stability": float(np.std(mean_stabs)) if mean_stabs else None,
        "std_min_stability": float(np.std(min_stabs)) if min_stabs else None,
    }
    return {"summary": summary, "per_file": rows}


def main() -> None:
    p = argparse.ArgumentParser(description="Mean / min BrickGPT stability on generated .ldr samples.")
    p.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "generated_ldr" / "baseline_ablation_16088",
        help="Root folder containing one subdirectory per experiment / condition.",
    )
    p.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Write full results JSON here (default: <root>/brickgpt_stability_eval.json).",
    )
    p.add_argument(
        "--only",
        type=str,
        nargs="*",
        default=None,
        help="If set, only run these subdirectory names under --root.",
    )
    args = p.parse_args()

    root: Path = args.root.expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.only:
        wanted = set(args.only)
        subdirs = [d for d in subdirs if d.name in wanted]

    out_path = args.output_json or (root / "brickgpt_stability_eval.json")
    report: dict = {"root": str(root), "experiments": {}}

    for exp in subdirs:
        report["experiments"][exp.name] = eval_folder(exp)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v["summary"] for k, v in report["experiments"].items()}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
