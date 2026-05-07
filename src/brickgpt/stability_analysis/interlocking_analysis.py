import numpy as np
from brickgpt.stability_analysis.connectivity_analysis import connectivity_score


def _rectangles_overlap(b1, b2) -> bool:
    return (b1.x < b2.x + b2.h and b1.x + b1.h > b2.x and
            b1.y < b2.y + b2.w and b1.y + b1.w > b2.y)


def interlocking_score(brick_structure) -> float:
    """
    Measures how well bricks interlock across layers.
    A brick is "interlocked" if it is supported by >= 2 distinct bricks
    on the layer below.

    Returns a score in [0, 1]. Higher = better interlocking.
    """
    bricks = brick_structure.bricks
    if len(bricks) <= 1:
        return 1.0

    by_layer = {}
    for b in bricks:
        by_layer.setdefault(b.z, []).append(b)

    interlocked = 0
    total_evaluated = 0

    for brick in bricks:
        if brick.z == 0:
            continue
        total_evaluated += 1

        below_layer = by_layer.get(brick.z - 1, [])
        supports = sum(1 for other in below_layer if _rectangles_overlap(brick, other))
        if supports >= 2:
            interlocked += 1

    return interlocked / max(total_evaluated, 1)


def seam_coverage_score(brick_structure) -> float:
    """
    Measures what fraction of horizontal seams between adjacent bricks
    on the same layer are bridged by bricks on the layer above.
    Higher = more structurally sound brick layout.

    Returns a score in [0, 1].
    """
    bricks = brick_structure.bricks
    world_dim = brick_structure.world_dim
    if len(bricks) <= 1:
        return 1.0

    by_layer = {}
    for b in bricks:
        by_layer.setdefault(b.z, []).append(b)

    total_seam_cells = 0
    covered_seam_cells = 0

    for z, layer_bricks in by_layer.items():
        above_bricks = by_layer.get(z + 1, [])
        if not above_bricks:
            continue

        # Build per-brick ownership map for this layer
        owner = np.full((world_dim, world_dim), -1, dtype=int)
        for idx, b in enumerate(layer_bricks):
            owner[b.x:b.x + b.h, b.y:b.y + b.w] = idx

        # Build occupancy map for above layer
        above_occ = np.zeros((world_dim, world_dim), dtype=bool)
        for ab in above_bricks:
            above_occ[ab.x:ab.x + ab.h, ab.y:ab.y + ab.w] = True

        # Find seam cells: occupied cells adjacent to a cell owned by a different brick
        for b_idx, b in enumerate(layer_bricks):
            for x in range(b.x, b.x + b.h):
                for y in range(b.y, b.y + b.w):
                    is_seam = False
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < world_dim and 0 <= ny < world_dim:
                            if owner[nx, ny] != -1 and owner[nx, ny] != b_idx:
                                is_seam = True
                                break
                    if is_seam:
                        total_seam_cells += 1
                        if above_occ[x, y]:
                            covered_seam_cells += 1

    return covered_seam_cells / max(total_seam_cells, 1)


def comprehensive_stability_score(brick_structure, use_gurobi=False) -> dict:
    """
    Compute a comprehensive set of stability metrics.
    Returns a dict with individual scores and a weighted composite.
    """
    result = {
        "n_bricks": len(brick_structure),
        "has_collisions": brick_structure.has_collisions(),
        "collision_voxels": int(np.sum(brick_structure.voxel_occupancy > 1)),
        "has_floating": brick_structure.has_floating_bricks(),
        "is_connected": False,
        "connectivity_ratio": 0.0,
        "interlocking_score": 0.0,
        "seam_coverage": 0.0,
        "physics_stable": False,
        "max_physics_score": float('inf'),
        "composite_score": 0.0,
    }

    if result["has_collisions"] or result["n_bricks"] == 0:
        return result

    try:
        scores = connectivity_score(brick_structure)
        result["is_connected"] = (scores.max() < 1)
        total_occ = max(int(np.sum(brick_structure.voxel_occupancy > 0)), 1)
        disconnected = int(np.sum(scores > 0))
        result["connectivity_ratio"] = 1.0 - disconnected / total_occ
    except Exception:
        pass

    result["interlocking_score"] = interlocking_score(brick_structure)
    result["seam_coverage"] = seam_coverage_score(brick_structure)

    if use_gurobi and not brick_structure.has_out_of_bounds_bricks():
        try:
            result["physics_stable"] = brick_structure.is_stable()
            phys_scores = brick_structure.stability_scores()
            result["max_physics_score"] = float(phys_scores.max())
        except Exception:
            pass

    result["composite_score"] = (
        0.3 * (1.0 if not result["has_collisions"] else 0.0) +
        0.2 * result["connectivity_ratio"] +
        0.25 * result["interlocking_score"] +
        0.15 * result["seam_coverage"] +
        0.1 * (1.0 if result["physics_stable"] else 0.0)
    )

    return result
