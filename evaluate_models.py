"""
Comprehensive evaluation script for BrickGPT-style LEGO generation models.

Supports multiple model backends:
  * causal_lm   : plain HuggingFace AutoModelForCausalLM + chat template
                  (works for our SFT / GRPO full-FT models and Qwen base)
  * brickgpt    : brickgpt.models.BrickGPT (supports rejection sampling,
                  physics-informed rollback, logit masking)

Metrics (per sample + aggregated):
  - parse_success              : whether the generated text parses into bricks
  - n_bricks                   : number of generated bricks
  - has_collisions             : any voxel with occupancy > 1
  - collision_voxels           : #voxels with occupancy > 1
  - out_of_bounds              : any brick outside the [0, world_dim) box
  - has_floating               : whether any brick is floating (no support)
  - is_connected               : whether the structure is a single connected piece
  - connectivity_ratio         : fraction of voxels that are structurally supported
  - interlocking_score         : fraction of upper-layer bricks supported by >=2 lower bricks
  - seam_coverage              : fraction of layer-internal seams covered by the layer above
  - physics_stable             : Gurobi stability check (only if --use_gurobi)
  - max_physics_score          : worst stability residual
  - iou_vs_ground_truth        : voxel IoU between generated structure and GT bricks
  - iou_vs_input_point_cloud   : voxel IoU between generated structure and input point cloud
                                 (mirrors BrickGPT's shape-matching evaluation)
  - composite_score            : weighted combination (see comprehensive_stability_score)

Usage (single model):
  python evaluate_models.py \
    --model_path finetuned_models/qwen_pc_grpo_full_ft \
    --model_type causal_lm \
    --run_name qwen_pc_grpo_full_ft \
    --dataset datasets_pc/test.jsonl \
    --max_samples 100 \
    --output_dir eval_results \
    --use_gurobi
"""
import argparse
import csv
import gc
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from brickgpt.data.brick_structure import BrickStructure  # noqa: E402
from brickgpt.stability_analysis.interlocking_analysis import (  # noqa: E402
    comprehensive_stability_score,
)


WORLD_DIM = 20
POINT_RE = re.compile(r"\((\d+),(\d+),(\d+)\)")


def parse_bricks(text: str) -> BrickStructure | None:
    """Parse generated text into a BrickStructure. Returns None on failure."""
    try:
        lines = [l.strip() for l in text.strip().split("\n") if l.strip().endswith(")")]
        if not lines:
            return None
        return BrickStructure.from_txt("\n".join(lines))
    except Exception:
        return None


def extract_point_cloud_voxels(user_prompt: str, world_dim: int = WORLD_DIM) -> np.ndarray | None:
    """
    Parse the '### Input Point Cloud:' block of a user prompt into a binary voxel grid.
    Returns a (world_dim, world_dim, world_dim) uint8 array, or None if no PC found.
    """
    if "Input Point Cloud" not in user_prompt:
        return None
    pc_section = user_prompt.split("Input Point Cloud:", 1)[1]
    grid = np.zeros((world_dim, world_dim, world_dim), dtype=np.uint8)
    found = False
    for match in POINT_RE.finditer(pc_section):
        x, y, z = (int(v) for v in match.groups())
        if 0 <= x < world_dim and 0 <= y < world_dim and 0 <= z < world_dim:
            grid[x, y, z] = 1
            found = True
    return grid if found else None


def voxel_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two binary voxel grids, padded to a common shape."""
    a = (a > 0).astype(np.float32)
    b = (b > 0).astype(np.float32)
    if a.shape != b.shape:
        shape = tuple(max(sa, sb) for sa, sb in zip(a.shape, b.shape))
        pa = np.zeros(shape, dtype=np.float32)
        pb = np.zeros(shape, dtype=np.float32)
        pa[: a.shape[0], : a.shape[1], : a.shape[2]] = a
        pb[: b.shape[0], : b.shape[1], : b.shape[2]] = b
        a, b = pa, pb
    intersection = float(np.sum(a * b))
    union = float(np.sum(np.clip(a + b, 0, 1)))
    if union <= 0:
        return 0.0
    return intersection / union


def ground_truth_structure(messages: list[dict]) -> BrickStructure | None:
    gt_text = next((m["content"] for m in messages if m["role"] == "assistant"), "")
    if not gt_text.strip():
        return None
    try:
        lines = [l.strip() for l in gt_text.strip().split("\n") if l.strip()]
        return BrickStructure.from_txt("\n".join(lines))
    except Exception:
        return None


def user_prompt(messages: list[dict]) -> str:
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    return ""


# ----------------------------------------------------------------------------
# Generation backends
# ----------------------------------------------------------------------------
class CausalLMGenerator:
    """Wraps an AutoModelForCausalLM with chat-template prompting."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str | None = None,
        attn_implementation: str = "eager",
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok_path = tokenizer_path or model_path
        print(f"Loading tokenizer from: {tok_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = getattr(torch, dtype)
        print(f"Loading causal LM: {model_path} (attn={attn_implementation}, dtype={dtype})")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt_messages: list[dict],
        max_new_tokens: int = 8192,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        gen_ids = self.model.generate(**inputs, **gen_kwargs)
        gen_ids = gen_ids[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def close(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class BrickGPTGenerator:
    """Uses brickgpt.models.BrickGPT with rejection sampling + rollback."""

    def __init__(
        self,
        model_path: str,
        use_logit_masking: bool = True,
        use_gurobi: bool = False,
        max_regenerations: int = 0,
        max_brick_rejections: int = 500,
        instruction_format: str = "point_cloud",
        temperature: float = 0.6,
    ):
        from brickgpt.models import BrickGPT, BrickGPTConfig

        cfg = BrickGPTConfig(
            model_name_or_path=model_path,
            world_dim=WORLD_DIM,
            use_logit_masking=use_logit_masking,
            use_gurobi=use_gurobi,
            max_regenerations=max_regenerations,
            max_brick_rejections=max_brick_rejections,
            instruction_format=instruction_format,
            temperature=temperature,
        )
        print(f"Loading BrickGPT: {model_path} (logit_mask={use_logit_masking}, "
              f"regen={max_regenerations}, reject={max_brick_rejections})")
        self.brickgpt = BrickGPT(cfg)

    @torch.no_grad()
    def generate_from_prompt(self, prompt_messages: list[dict], **_ignored) -> str:
        user_msg = next((m["content"] for m in prompt_messages if m["role"] == "user"), "")
        caption = user_msg
        if "Input Point Cloud:" in user_msg:
            caption = user_msg.split("Input Point Cloud:", 1)[1].strip()
        elif "### Input:" in user_msg:
            caption = user_msg.split("### Input:", 1)[1].strip()
        output = self.brickgpt(caption)
        return output["bricks"].to_txt()

    def close(self):
        del self.brickgpt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# Main evaluation loop
# ----------------------------------------------------------------------------
def evaluate_sample(
    sample: dict,
    response_text: str,
    use_gurobi: bool,
) -> dict:
    messages = sample["messages"]
    prompt = user_prompt(messages)
    pc_voxels = extract_point_cloud_voxels(prompt)
    gt_structure = ground_truth_structure(messages)

    result: dict = {
        "parse_success": False,
        "n_bricks": 0,
        "has_collisions": True,
        "collision_voxels": 0,
        "out_of_bounds": False,
        "has_floating": True,
        "is_connected": False,
        "connectivity_ratio": 0.0,
        "interlocking_score": 0.0,
        "seam_coverage": 0.0,
        "physics_stable": False,
        "max_physics_score": float("inf"),
        "composite_score": 0.0,
        "iou_vs_ground_truth": 0.0,
        "iou_vs_input_point_cloud": 0.0,
        "response_tokens": len(response_text),
    }

    structure = parse_bricks(response_text)
    if structure is None or len(structure) == 0:
        return result

    result["parse_success"] = True
    try:
        result["out_of_bounds"] = bool(structure.has_out_of_bounds_bricks())
    except Exception:
        pass

    try:
        stab = comprehensive_stability_score(structure, use_gurobi=use_gurobi)
        for key in [
            "n_bricks",
            "has_collisions",
            "collision_voxels",
            "has_floating",
            "is_connected",
            "connectivity_ratio",
            "interlocking_score",
            "seam_coverage",
            "physics_stable",
            "max_physics_score",
            "composite_score",
        ]:
            if key in stab:
                result[key] = stab[key]
    except Exception:
        pass

    if gt_structure is not None:
        try:
            result["iou_vs_ground_truth"] = voxel_iou(
                structure.voxel_occupancy, gt_structure.voxel_occupancy
            )
        except Exception:
            pass

    if pc_voxels is not None:
        try:
            result["iou_vs_input_point_cloud"] = voxel_iou(
                structure.voxel_occupancy, pc_voxels
            )
        except Exception:
            pass

    for key, val in list(result.items()):
        if isinstance(val, (np.bool_,)):
            result[key] = bool(val)
        elif isinstance(val, (np.floating,)):
            result[key] = float(val)
        elif isinstance(val, (np.integer,)):
            result[key] = int(val)
        elif isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            result[key] = None if np.isnan(val) else ("inf" if val > 0 else "-inf")

    return result


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n_samples": 0}

    parse_ok = [r for r in results if r["parse_success"]]
    collision_free = [r for r in parse_ok if not r["has_collisions"]]
    in_bounds = [r for r in parse_ok if not r["out_of_bounds"]]
    clean = [r for r in parse_ok if (not r["has_collisions"]) and (not r["out_of_bounds"])]

    def mean(key, pool):
        vals = [r[key] for r in pool if isinstance(r[key], (int, float))]
        return float(np.mean(vals)) if vals else 0.0

    def rate(key, pool, truthy=True):
        if not pool:
            return 0.0
        hits = sum(1 for r in pool if bool(r[key]) == truthy)
        return hits / len(pool)

    summary = {
        "n_samples": n,
        "parse_success_rate": len(parse_ok) / n,
        "collision_free_rate_of_parsed": (len(collision_free) / max(len(parse_ok), 1)),
        "collision_free_rate_of_total": len(collision_free) / n,
        "in_bounds_rate_of_parsed": len(in_bounds) / max(len(parse_ok), 1),
        "clean_rate_of_total": len(clean) / n,
        "mean_n_bricks": mean("n_bricks", parse_ok),
        "mean_collision_voxels": mean("collision_voxels", parse_ok),
        "mean_iou_vs_ground_truth": mean("iou_vs_ground_truth", parse_ok),
        "mean_iou_vs_input_point_cloud": mean("iou_vs_input_point_cloud", parse_ok),
        "mean_connectivity_ratio": mean("connectivity_ratio", parse_ok),
        "is_connected_rate_of_parsed": rate("is_connected", parse_ok),
        "mean_interlocking_score": mean("interlocking_score", parse_ok),
        "mean_seam_coverage": mean("seam_coverage", parse_ok),
        "mean_composite_score": mean("composite_score", parse_ok),
        "physics_stable_rate_of_clean": rate("physics_stable", clean),
        "physics_stable_rate_of_total": sum(1 for r in results if r.get("physics_stable")) / n,
    }
    return summary


def save_outputs(
    run_name: str,
    output_dir: str,
    results: list[dict],
    summary: dict,
    gen_texts: list[str] | None,
    args_dict: dict,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / f"{run_name}_per_sample.csv"
    if results:
        keys = sorted({k for r in results for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

    summary_path = out / f"{run_name}_summary.json"
    payload = {"run_name": run_name, "args": args_dict, "summary": summary}
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)

    if gen_texts is not None:
        gen_dir = out / f"{run_name}_generations"
        gen_dir.mkdir(exist_ok=True)
        for i, txt in enumerate(gen_texts):
            (gen_dir / f"sample_{i:04d}.txt").write_text(txt)

    print(f"\nSaved results:")
    print(f"  per-sample CSV : {csv_path}")
    print(f"  summary JSON   : {summary_path}")


def print_summary(run_name: str, summary: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"SUMMARY  [{run_name}]")
    print(f"{'=' * 72}")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<40s} {v:.4f}")
        else:
            print(f"  {k:<40s} {v}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True,
                        help="HF repo id or local path of the model to evaluate.")
    parser.add_argument("--model_type", choices=["causal_lm", "brickgpt"], default="causal_lm")
    parser.add_argument("--tokenizer_path", default=None,
                        help="Defaults to --model_path. Useful when the checkpoint "
                             "directory lacks tokenizer files.")
    parser.add_argument("--dataset", default="datasets_pc/test.jsonl")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--output_dir", default="eval_results")
    parser.add_argument("--run_name", required=True,
                        help="Identifier used as filename prefix for outputs.")
    parser.add_argument("--use_gurobi", action="store_true",
                        help="Enable Gurobi-based physics stability check.")
    parser.add_argument("--save_generations", action="store_true")

    # Causal-LM options
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--dtype", default="bfloat16")

    # BrickGPT-mode options
    parser.add_argument("--brickgpt_use_logit_masking", action="store_true",
                        help="Enable BrickGPT logit masking.")
    parser.add_argument("--brickgpt_max_regenerations", type=int, default=0)
    parser.add_argument("--brickgpt_max_brick_rejections", type=int, default=0)
    parser.add_argument("--brickgpt_instruction_format", default="point_cloud",
                        choices=["point_cloud", "brickgpt", "zero_shot", "few_shot"])
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    args_dict = vars(args).copy()

    print(f"[eval] run_name  : {args.run_name}")
    print(f"[eval] model_path: {args.model_path}")
    print(f"[eval] model_type: {args.model_type}")
    print(f"[eval] dataset   : {args.dataset}")
    print(f"[eval] samples   : max={args.max_samples} start={args.start_index}")
    print(f"[eval] use_gurobi: {args.use_gurobi}")

    if args.model_type == "causal_lm":
        generator = CausalLMGenerator(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            attn_implementation=args.attn_implementation,
            dtype=args.dtype,
        )
    else:
        generator = BrickGPTGenerator(
            model_path=args.model_path,
            use_logit_masking=args.brickgpt_use_logit_masking,
            use_gurobi=False,
            max_regenerations=args.brickgpt_max_regenerations,
            max_brick_rejections=args.brickgpt_max_brick_rejections,
            instruction_format=args.brickgpt_instruction_format,
            temperature=args.temperature,
        )

    results: list[dict] = []
    gen_texts: list[str] = [] if args.save_generations else None
    t_start = time.time()

    with open(args.dataset) as f:
        lines = f.readlines()
    lines = lines[args.start_index : args.start_index + args.max_samples]
    print(f"[eval] loaded {len(lines)} samples from {args.dataset}")

    for idx, raw in enumerate(lines):
        sample = json.loads(raw)
        prompt_messages = [m for m in sample["messages"] if m["role"] != "assistant"]

        t0 = time.time()
        try:
            if args.model_type == "causal_lm":
                response = generator.generate(
                    prompt_messages,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            else:
                response = generator.generate_from_prompt(prompt_messages)
        except Exception as exc:
            print(f"[eval][{idx + 1}/{len(lines)}] generation error: {exc}")
            traceback.print_exc()
            response = ""
        gen_time = time.time() - t0

        metrics = evaluate_sample(sample, response, use_gurobi=args.use_gurobi)
        metrics["sample_index"] = args.start_index + idx
        metrics["gen_seconds"] = round(gen_time, 2)
        results.append(metrics)
        if gen_texts is not None:
            gen_texts.append(response)

        status = (
            "PARSE_FAIL" if not metrics["parse_success"]
            else "OK" if (not metrics["has_collisions"] and not metrics["out_of_bounds"])
            else "COLLIDE" if metrics["has_collisions"]
            else "OOB"
        )
        print(
            f"[eval][{idx + 1:03d}/{len(lines):03d}] {status:10s} "
            f"bricks={metrics['n_bricks']:4d} "
            f"iou_gt={metrics['iou_vs_ground_truth']:.3f} "
            f"iou_pc={metrics['iou_vs_input_point_cloud']:.3f} "
            f"conn={metrics['connectivity_ratio']:.3f} "
            f"interlock={metrics['interlocking_score']:.3f} "
            f"seam={metrics['seam_coverage']:.3f} "
            f"phys={int(bool(metrics['physics_stable']))} "
            f"t={gen_time:.1f}s"
        )

    elapsed = time.time() - t_start
    summary = aggregate(results)
    summary["elapsed_seconds"] = round(elapsed, 2)
    summary["samples_per_second"] = round(len(results) / max(elapsed, 1e-6), 4)

    print_summary(args.run_name, summary)
    save_outputs(args.run_name, args.output_dir, results, summary, gen_texts, args_dict)

    try:
        generator.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
