"""
Convert SFT dataset (datasets_pc/) to GRPO format.
GRPO needs a 'prompt' column (list of message dicts) and extra columns for reward kwargs.
We store the ground truth voxels as a base64-encoded compressed numpy array.
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from brickgpt.data.brick_structure import BrickStructure


def encode_voxels(voxel_occupancy: np.ndarray) -> str:
    binary = (voxel_occupancy > 0).astype(np.uint8)
    buf = io.BytesIO()
    np.save(buf, binary)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    input_dir = Path("datasets_pc")
    output_dir = Path("datasets_pc")

    for split in ["train", "test"]:
        input_path = input_dir / f"{split}.jsonl"
        output_path = output_dir / f"{split}_grpo.jsonl"
        if not input_path.exists():
            print(f"Skipping {split}: {input_path} not found")
            continue

        count = 0
        skipped = 0
        with open(input_path) as fin, open(output_path, "w") as fout:
            for line in fin:
                sample = json.loads(line)
                messages = sample["messages"]

                prompt = [m for m in messages if m["role"] != "assistant"]
                gt_text = next((m["content"] for m in messages if m["role"] == "assistant"), "")

                if not gt_text.strip():
                    skipped += 1
                    continue

                try:
                    lines = [l for l in gt_text.strip().split("\n") if l.strip()]
                    structure = BrickStructure.from_txt("\n".join(lines))
                    target_voxels = encode_voxels(structure.voxel_occupancy)
                except Exception:
                    skipped += 1
                    continue

                grpo_sample = {
                    "prompt": prompt,
                    "target_voxels": target_voxels,
                }
                fout.write(json.dumps(grpo_sample) + "\n")
                count += 1

        print(f"{split}: converted {count} samples, skipped {skipped} -> {output_path}")


if __name__ == "__main__":
    main()
