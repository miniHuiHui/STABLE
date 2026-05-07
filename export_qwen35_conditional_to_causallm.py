"""Export the text backbone of a Qwen3.5 ConditionalGeneration checkpoint as CausalLM."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM


TEXT_WEIGHT_PREFIX = "model.language_model.language_model.language_model."
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
]


def build_causal_config(source_config: dict):
    text_config = dict(source_config["text_config"])
    text_config.pop("model_type", None)
    config = AutoConfig.for_model("qwen3_5_text", **text_config)
    config.architectures = ["Qwen3_5ForCausalLM"]
    config.tie_word_embeddings = source_config.get("tie_word_embeddings", True)
    config.eos_token_id = source_config.get("eos_token_id", config.eos_token_id)
    config.pad_token_id = source_config.get("pad_token_id", config.pad_token_id)
    config.use_cache = source_config.get("use_cache", True)
    return config


def extract_text_state_dict(weights_path: Path):
    state_dict = {}
    skipped_keys = []

    with safe_open(str(weights_path), framework="pt") as f:
        for key in f.keys():
            if not key.startswith(TEXT_WEIGHT_PREFIX):
                skipped_keys.append(key)
                continue

            remapped_key = "model." + key[len(TEXT_WEIGHT_PREFIX) :]
            state_dict[remapped_key] = f.get_tensor(key)

    return state_dict, skipped_keys


def copy_tokenizer_assets(source_dir: Path, output_dir: Path):
    for filename in TOKENIZER_FILES:
        src = source_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        default="finetuned_models/qwen35_sft_full_ft_run1",
        help="Path to the ConditionalGeneration checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        default="finetuned_models/qwen35_sft_full_ft_run1_causallm",
        help="Path to save the exported CausalLM checkpoint",
    )
    parser.add_argument(
        "--max_shard_size",
        default="5GB",
        help="Max shard size passed to save_pretrained",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = source_dir / "config.json"
    weights_path = source_dir / "model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing safetensors weights: {weights_path}")

    with config_path.open() as f:
        source_config = json.load(f)

    config = build_causal_config(source_config)

    print(f"Loading text weights from {weights_path}")
    state_dict, skipped_keys = extract_text_state_dict(weights_path)
    print(f"Loaded {len(state_dict)} text weights")
    print(f"Skipped {len(skipped_keys)} non-text weights")

    print("Building CausalLM model on CPU")
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    missing_keys = sorted(missing_keys)
    unexpected_keys = sorted(unexpected_keys)

    allowed_missing = {"lm_head.weight"}
    if set(missing_keys) - allowed_missing:
        raise RuntimeError(f"Unexpected missing keys: {missing_keys}")
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys after remap: {unexpected_keys[:20]}")

    # Restore shared output/input embeddings for tied-word-embedding models.
    model.tie_weights()

    print(f"Saving exported CausalLM checkpoint to {output_dir}")
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    copy_tokenizer_assets(source_dir, output_dir)

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "loaded_text_weights": len(state_dict),
        "skipped_non_text_weights": len(skipped_keys),
        "missing_keys_after_load": missing_keys,
    }
    with (output_dir / "export_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("Export complete")


if __name__ == "__main__":
    main()
