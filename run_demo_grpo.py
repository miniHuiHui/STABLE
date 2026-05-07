"""
Generate test samples from a full fine-tuned GRPO checkpoint.
"""
import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from brickgpt.data.brick_structure import BrickStructure


BRICK_LINE_RE = re.compile(r"^\s*(\d+)x(\d+)\s+\(\d+,\d+,\d+\)\s*$")


def load_model(grpo_checkpoint: str):
    # Eager attention is slowest but avoids CUDA illegal-access issues on some driver/GPU combos.
    model = AutoModelForCausalLM.from_pretrained(
        grpo_checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map="auto",
    )
    model.eval()
    return model


def append_constraint_to_messages(prompt_messages: list[dict], constraint: str | None) -> list[dict]:
    if not constraint:
        return prompt_messages
    patched_messages: list[dict] = []
    for m in prompt_messages:
        if m.get("role") == "user":
            patched = dict(m)
            patched["content"] = f"{m.get('content', '').rstrip()}\n\n{constraint.strip()}\n"
            patched_messages.append(patched)
        else:
            patched_messages.append(m)
    return patched_messages


def filter_brick_lines_by_size(response: str, only_size: str | None) -> tuple[str, int]:
    if not only_size:
        return response, 0
    if "x" not in only_size:
        raise ValueError(f"Invalid --only_brick_size format: {only_size}. Expected e.g. 1x1")
    h0, w0 = only_size.lower().split("x", 1)
    target_h, target_w = int(h0), int(w0)

    kept: list[str] = []
    dropped = 0
    for raw_line in response.split("\n"):
        line = raw_line.strip()
        if not line.endswith(")"):
            continue
        match = BRICK_LINE_RE.match(line)
        if match is None:
            continue
        h, w = int(match.group(1)), int(match.group(2))
        if (h, w) == (target_h, target_w):
            kept.append(line)
        else:
            dropped += 1
    filtered = "\n".join(kept)
    return filtered, dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grpo_checkpoint",
        default="finetuned_models/qwen_pc_grpo_full_ft",
        help="Path to full fine-tuned GRPO weights (merged dir or checkpoint-* folder)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-2B",
        help="Tokenizer source if the checkpoint directory does not include tokenizer files",
    )
    parser.add_argument("--dataset", default="datasets_pc/test.jsonl")
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Skip the first N lines of the JSONL (for resuming runs)",
    )
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stochastic decoding; default False (greedy), matching evaluate_models.py",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Defaults to demo_grpo_<checkpoint_name>_outputs",
    )
    parser.add_argument(
        "--only_brick_size",
        default=None,
        help="Keep only generated bricks of this size in output/export, e.g. 1x1.",
    )
    parser.add_argument(
        "--append_user_constraint",
        default=None,
        help="Extra natural-language instruction appended to each user prompt.",
    )
    args = parser.parse_args()

    ck_name = os.path.basename(os.path.normpath(args.grpo_checkpoint))
    out_dir = args.output_dir or f"demo_grpo_{ck_name}_outputs"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.grpo_checkpoint)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = 0.9

    with open(args.dataset, "r") as f:
        for _ in range(args.start_index):
            if not f.readline():
                print(f"Warning: dataset has fewer than start_index={args.start_index} lines")
                return
        for i in range(args.num_samples):
            line = f.readline()
            if not line:
                break
            sample = json.loads(line)
            messages = sample["messages"]
            prompt_messages = [m for m in messages if m["role"] != "assistant"]
            prompt_messages = append_constraint_to_messages(
                prompt_messages, args.append_user_constraint
            )
            ground_truth = next(
                (m["content"] for m in messages if m["role"] == "assistant"), None
            )

            idx = args.start_index + i + 1
            text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            print(f"Generating sample {idx} (batch {i + 1}/{args.num_samples}, start_index={args.start_index})...")
            with torch.no_grad():
                generated_ids = model.generate(**model_inputs, **gen_kwargs)

            gen_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            filtered_response, dropped_non_target = filter_brick_lines_by_size(
                response, args.only_brick_size
            )

            out_txt = os.path.join(out_dir, f"sample_{idx:04d}_gen_raw.txt")
            with open(out_txt, "w") as out_f:
                out_f.write(response.strip())
            with open(os.path.join(out_dir, f"sample_{idx:04d}_gen.txt"), "w") as out_f:
                out_f.write(filtered_response.strip())

            meta = {
                "sample_index": args.start_index + i + 1,
                "line_index_0based": args.start_index + i,
                "grpo_checkpoint": args.grpo_checkpoint,
                "only_brick_size": args.only_brick_size,
                "append_user_constraint": args.append_user_constraint,
                "dropped_non_target_bricks": dropped_non_target,
                "ground_truth_preview": (ground_truth[:2000] if ground_truth else "")
                + ("..." if ground_truth and len(ground_truth) > 2000 else ""),
            }
            with open(os.path.join(out_dir, f"sample_{idx:04d}_meta.json"), "w") as meta_fp:
                json.dump(meta, meta_fp, indent=2, ensure_ascii=False)

            if ground_truth:
                gt_path = os.path.join(out_dir, f"sample_{idx:04d}_ground_truth.txt")
                with open(gt_path, "w") as gt_fp:
                    gt_fp.write(ground_truth.strip())

            try:
                lines = [
                    x for x in filtered_response.strip().split("\n")
                    if x.strip().endswith(")")
                ]
                if lines:
                    s = BrickStructure.from_txt("\n".join(lines))
                    with open(os.path.join(out_dir, f"sample_{idx:04d}_gen.ldr"), "w") as out_f:
                        out_f.write(s.to_ldr())
            except Exception as e:
                print(f"  LDR skip sample {idx}: {e}")

    print(f"Done. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
