"""
Evaluate a fine-tuned model on the test set.
Computes: collision rate, IoU, interlocking score, seam coverage, connectivity, physics stability.

Usage:
    conda run -n qwen_brickgpt_2 python evaluate_model.py --checkpoint <path> [--use_gurobi]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, "src")
from brickgpt.data.brick_structure import BrickStructure
from brickgpt.stability_analysis.interlocking_analysis import comprehensive_stability_score


def parse_bricks(text: str) -> BrickStructure | None:
    try:
        lines = [l.strip() for l in text.strip().split("\n") if l.strip().endswith(")")]
        if not lines:
            return None
        return BrickStructure.from_txt("\n".join(lines))
    except Exception:
        return None


def compute_iou(gen_structure: BrickStructure, gt_text: str) -> float:
    try:
        gt_lines = [l.strip() for l in gt_text.strip().split("\n") if l.strip()]
        gt_structure = BrickStructure.from_txt("\n".join(gt_lines))
        gen_voxels = (gen_structure.voxel_occupancy > 0).astype(float)
        gt_voxels = (gt_structure.voxel_occupancy > 0).astype(float)
        intersection = np.sum(gen_voxels * gt_voxels)
        union = np.sum(np.clip(gen_voxels + gt_voxels, 0, 1))
        return intersection / max(union, 1e-8)
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--checkpoint", required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--dataset", default="datasets_pc/test.jsonl")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--use_gurobi", action="store_true")
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    print(f"Loading LoRA adapter: {args.checkpoint}")
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()

    results = []
    with open(args.dataset) as f:
        for i, line in enumerate(f):
            if i >= args.max_samples:
                break
            sample = json.loads(line)
            messages = sample["messages"]
            prompt_messages = [m for m in messages if m["role"] != "assistant"]
            gt_text = next((m["content"] for m in messages if m["role"] == "assistant"), "")

            text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=8192, do_sample=False)
            gen_ids = gen_ids[0][inputs.input_ids.shape[1]:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)

            structure = parse_bricks(response)
            if structure is None or len(structure) == 0:
                results.append({
                    "sample": i, "n_bricks": 0, "parse_error": True,
                    "collision_voxels": 0, "iou": 0.0, "interlocking_score": 0.0,
                    "seam_coverage": 0.0, "connectivity_ratio": 0.0,
                    "physics_stable": False, "composite_score": 0.0,
                })
                print(f"[{i+1}] PARSE ERROR")
                continue

            iou = compute_iou(structure, gt_text)
            scores = comprehensive_stability_score(structure, use_gurobi=args.use_gurobi)
            scores["sample"] = i
            scores["iou"] = iou
            scores["parse_error"] = False
            results.append(scores)

            status = "OK" if not scores["has_collisions"] else f"COLLISION({scores['collision_voxels']})"
            print(f"[{i+1}] {status} | IoU={iou:.3f} | interlock={scores['interlocking_score']:.3f} "
                  f"| conn={scores['connectivity_ratio']:.3f} | composite={scores['composite_score']:.3f}")

    # Print summary
    n = len(results)
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY ({n} samples from {args.checkpoint})")
    print(f"{'='*60}")
    parse_ok = [r for r in results if not r.get("parse_error")]
    collision_free = [r for r in parse_ok if not r.get("has_collisions")]

    print(f"Parse success rate: {len(parse_ok)}/{n} ({100*len(parse_ok)/max(n,1):.1f}%)")
    print(f"Collision-free rate: {len(collision_free)}/{len(parse_ok)} ({100*len(collision_free)/max(len(parse_ok),1):.1f}%)")

    if parse_ok:
        avg_iou = np.mean([r["iou"] for r in parse_ok])
        avg_interlock = np.mean([r["interlocking_score"] for r in parse_ok])
        avg_seam = np.mean([r["seam_coverage"] for r in parse_ok])
        avg_conn = np.mean([r["connectivity_ratio"] for r in parse_ok])
        avg_composite = np.mean([r["composite_score"] for r in parse_ok])
        print(f"Mean IoU: {avg_iou:.4f}")
        print(f"Mean interlocking: {avg_interlock:.4f}")
        print(f"Mean seam coverage: {avg_seam:.4f}")
        print(f"Mean connectivity: {avg_conn:.4f}")
        print(f"Mean composite: {avg_composite:.4f}")

        if args.use_gurobi:
            physics_stable = [r for r in collision_free if r.get("physics_stable")]
            print(f"Physics-stable rate: {len(physics_stable)}/{len(collision_free)}")

    if args.output_csv:
        csv_path = Path(args.output_csv)
        keys = results[0].keys() if results else []
        with open(csv_path, "w", newline="") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nDetailed results saved to {csv_path}")


if __name__ == "__main__":
    main()
