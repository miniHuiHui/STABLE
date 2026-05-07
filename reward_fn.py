"""
Reward functions for GRPO training.
Each function has signature: (prompts, completions, completion_ids, **kwargs) -> list[float]

Optimized: parses each completion only once via a shared cache.
"""
import base64
import io
import sys
import threading

import numpy as np

sys.path.insert(0, "src")
from brickgpt.data.brick_structure import BrickStructure
from brickgpt.stability_analysis.connectivity_analysis import connectivity_score
from brickgpt.stability_analysis.interlocking_analysis import interlocking_score

_parse_cache: dict[str, BrickStructure | None] = {}
_cache_lock = threading.Lock()
_MAX_CACHE = 256


def _clear_cache_if_large():
    if len(_parse_cache) > _MAX_CACHE:
        _parse_cache.clear()


def decode_voxels(encoded: str) -> np.ndarray:
    buf = io.BytesIO(base64.b64decode(encoded))
    return np.load(buf)


def _extract_text(completion) -> str:
    if isinstance(completion, list):
        return completion[0]["content"]
    return completion


def _parse_bricks(text: str) -> BrickStructure | None:
    with _cache_lock:
        _clear_cache_if_large()
        if text in _parse_cache:
            return _parse_cache[text]
    try:
        lines = [l.strip() for l in text.strip().split("\n") if l.strip().endswith(")")]
        if not lines:
            result = None
        else:
            result = BrickStructure.from_txt("\n".join(lines))
    except Exception:
        result = None
    with _cache_lock:
        _parse_cache[text] = result
    return result


def collision_reward(prompts, completions, completion_ids, **kwargs):
    """Penalize collisions: -2 per colliding voxel, clamped to [-10, 0]."""
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        structure = _parse_bricks(text)
        if structure is None or len(structure) == 0:
            rewards.append(-10.0)
            continue
        collision_count = int(np.sum(structure.voxel_occupancy > 1))
        rewards.append(max(-10.0, -2.0 * collision_count))
    return rewards


def shape_reward(prompts, completions, completion_ids, **kwargs):
    """IoU between generated voxels and target voxels. Range [0, 5]."""
    target_voxels_list = kwargs.get("target_voxels", [])
    rewards = []
    for i, completion in enumerate(completions):
        text = _extract_text(completion)
        structure = _parse_bricks(text)
        if structure is None or len(structure) == 0:
            rewards.append(0.0)
            continue

        gen_voxels = (structure.voxel_occupancy > 0).astype(np.float32)

        if i < len(target_voxels_list) and target_voxels_list[i]:
            try:
                target = decode_voxels(target_voxels_list[i]).astype(np.float32)
                if target.shape != gen_voxels.shape:
                    padded = np.zeros_like(gen_voxels)
                    slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(target.shape, gen_voxels.shape))
                    padded[slices] = target[slices]
                    target = padded
                intersection = np.sum(gen_voxels * target)
                union = np.sum(np.clip(gen_voxels + target, 0, 1))
                iou = intersection / max(union, 1e-8)
                rewards.append(5.0 * iou)
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards


def interlocking_reward(prompts, completions, completion_ids, **kwargs):
    """Interlocking reward. Only applied when structure has no collisions."""
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        structure = _parse_bricks(text)
        if structure is None or len(structure) == 0:
            rewards.append(0.0)
            continue

        if structure.has_collisions() or structure.has_out_of_bounds_bricks():
            rewards.append(0.0)
            continue

        try:
            rewards.append(3.0 * interlocking_score(structure))
        except Exception:
            rewards.append(0.0)
    return rewards


def connectivity_reward(prompts, completions, completion_ids, **kwargs):
    """Connectivity reward. Only applied when structure has no collisions."""
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        structure = _parse_bricks(text)
        if structure is None or len(structure) == 0:
            rewards.append(0.0)
            continue

        if structure.has_collisions() or structure.has_out_of_bounds_bricks():
            rewards.append(0.0)
            continue

        try:
            scores = connectivity_score(structure)
            total_occ = max(int(np.sum(structure.voxel_occupancy > 0)), 1)
            disconnected = int(np.sum(scores > 0))
            rewards.append(2.0 * (1.0 - disconnected / total_occ))
        except Exception:
            rewards.append(0.0)
    return rewards
