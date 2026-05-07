"""Export a full-ft Qwen3.5 checkpoint to the official repo layout."""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


SOURCE_TEXT_PREFIX = "model.language_model.language_model.language_model."
SOURCE_VISUAL_PREFIX = "model.language_model.visual."
TARGET_TEXT_PREFIX = "model.language_model."
TARGET_VISUAL_PREFIX = "model.visual."
MTP_PREFIX = "mtp."

OFFICIAL_METADATA_FILES = [
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
]
SOURCE_OPTIONAL_FILES = [
    "generation_config.json",
]


def remap_source_weights(source_weights_path: Path) -> dict[str, object]:
    remapped: dict[str, object] = {}
    unexpected: list[str] = []

    with safe_open(str(source_weights_path), framework="pt") as f:
        for key in f.keys():
            if key.startswith(SOURCE_TEXT_PREFIX):
                new_key = TARGET_TEXT_PREFIX + key[len(SOURCE_TEXT_PREFIX) :]
            elif key.startswith(SOURCE_VISUAL_PREFIX):
                new_key = TARGET_VISUAL_PREFIX + key[len(SOURCE_VISUAL_PREFIX) :]
            else:
                unexpected.append(key)
                continue
            remapped[new_key] = f.get_tensor(key)

    if unexpected:
        raise RuntimeError(f"Unexpected source keys: {unexpected[:20]}")
    return remapped


def load_official_mtp_weights(official_weights_path: Path) -> dict[str, object]:
    mtp_weights: dict[str, object] = {}
    with safe_open(str(official_weights_path), framework="pt") as f:
        for key in f.keys():
            if key.startswith(MTP_PREFIX):
                mtp_weights[key] = f.get_tensor(key)
    if not mtp_weights:
        raise RuntimeError("No official mtp.* weights found to preserve official layout")
    return mtp_weights


def write_single_shard_checkpoint(output_dir: Path, state_dict: dict[str, object]) -> None:
    output_weights = output_dir / "model.safetensors-00001-of-00001.safetensors"
    save_file(state_dict, str(output_weights))

    total_size = sum(tensor.nelement() * tensor.element_size() for tensor in state_dict.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {key: output_weights.name for key in state_dict},
    }
    with (output_dir / "model.safetensors.index.json").open("w") as f:
        json.dump(index, f, indent=2, sort_keys=False)


def copy_official_metadata(official_dir: Path, output_dir: Path) -> None:
    for filename in OFFICIAL_METADATA_FILES:
        src = official_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing official metadata file: {src}")
        shutil.copy2(src, output_dir / filename)


def copy_source_optional_files(source_dir: Path, output_dir: Path) -> None:
    for filename in SOURCE_OPTIONAL_FILES:
        src = source_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)


def write_processor_files(source_dir: Path, output_dir: Path) -> None:
    processor_path = source_dir / "processor_config.json"
    if not processor_path.exists():
        return

    processor_config = json.loads(processor_path.read_text())
    with (output_dir / "processor_config.json").open("w") as f:
        json.dump(processor_config, f, indent=2)

    image_processor_config = processor_config.get("image_processor")
    if image_processor_config:
        with (output_dir / "preprocessor_config.json").open("w") as f:
            json.dump(image_processor_config, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        default="finetuned_models/qwen35_sft_full_ft_run1",
        help="Path to the saved full-ft checkpoint directory",
    )
    parser.add_argument(
        "--official_dir",
        default=".hf_qwen35_official",
        help="Path to a downloaded official Qwen3.5 repo snapshot",
    )
    parser.add_argument(
        "--output_dir",
        default="finetuned_models/qwen35_sft_full_ft_run1_official",
        help="Path to save the official-format export",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    official_dir = Path(args.official_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_weights_path = source_dir / "model.safetensors"
    official_weights_path = official_dir / "model.safetensors-00001-of-00001.safetensors"
    if not source_weights_path.exists():
        raise FileNotFoundError(f"Missing source weights: {source_weights_path}")
    if not official_weights_path.exists():
        raise FileNotFoundError(f"Missing official weights: {official_weights_path}")

    print(f"Remapping source weights from {source_weights_path}")
    state_dict = remap_source_weights(source_weights_path)

    print(f"Loading official MTP weights from {official_weights_path}")
    mtp_weights = load_official_mtp_weights(official_weights_path)
    overlap = set(state_dict).intersection(mtp_weights)
    if overlap:
        raise RuntimeError(f"Unexpected overlap with official mtp weights: {sorted(overlap)[:20]}")
    state_dict.update(mtp_weights)

    print(f"Writing official-format checkpoint to {output_dir}")
    write_single_shard_checkpoint(output_dir, state_dict)
    copy_official_metadata(official_dir, output_dir)
    copy_source_optional_files(source_dir, output_dir)
    write_processor_files(source_dir, output_dir)
    print("Official-format export complete")


if __name__ == "__main__":
    main()
