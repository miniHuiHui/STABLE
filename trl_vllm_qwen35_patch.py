"""Patch TRL/vLLM interop issues for Qwen3.5 GRPO training."""

import logging


logger = logging.getLogger(__name__)


VLLM_PREFIX_MAPS: dict[str, dict[str, str]] = {
    "Qwen3_5ForConditionalGeneration": {
        "model.visual.": "visual.",
        "lm_head.": "language_model.lm_head.",
        "model.language_model.": "language_model.model.",
    },
    "Qwen3VLForConditionalGeneration": {
        "model.visual.": "visual.",
        "lm_head.": "language_model.lm_head.",
        "model.language_model.": "language_model.model.",
    },
    "Qwen3_5ForCausalLM": {
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
    },
}


def patch_vllm_generation_qwen35_prefixes(trainer) -> None:
    """Patch TRL's vLLM name mapping for known Qwen3.5/Qwen3-VL prefixes."""

    vllm_generation = getattr(trainer, "vllm_generation", None)
    model = getattr(trainer, "model", None)
    if vllm_generation is None or model is None:
        return

    if getattr(vllm_generation, "_brickgpt_qwen35_prefix_patch", False):
        return

    model_config = getattr(model, "config", None)
    model_architectures = getattr(model_config, "architectures", []) or []
    hf_to_vllm_prefix = None
    for arch in model_architectures:
        if arch in VLLM_PREFIX_MAPS:
            hf_to_vllm_prefix = VLLM_PREFIX_MAPS[arch]
            break

    # Some Qwen3.5 checkpoints do not expose `config.architectures` on the HF-side model
    # even though the actual parameter names clearly identify the text backbone layout.
    if hf_to_vllm_prefix is None:
        sample_param_names = []
        for idx, (name, _) in enumerate(model.named_parameters()):
            sample_param_names.append(name)
            if idx >= 15:
                break
        if any(name.startswith("model.language_model.") or name.startswith("model.visual.") for name in sample_param_names):
            hf_to_vllm_prefix = VLLM_PREFIX_MAPS["Qwen3_5ForConditionalGeneration"]
            model_architectures = model_architectures or ["inferred:Qwen3_5ForConditionalGeneration"]
        elif any(name.startswith("model.") or name.startswith("lm_head.") for name in sample_param_names):
            hf_to_vllm_prefix = VLLM_PREFIX_MAPS["Qwen3_5ForCausalLM"]
            model_architectures = model_architectures or ["inferred:Qwen3_5ForCausalLM"]

    if hf_to_vllm_prefix is None:
        return

    original_fix = getattr(vllm_generation, "_fix_param_name_to_vllm", None)
    if original_fix is None:
        return

    def _fix_param_name_to_vllm(name: str, extra_prefixes: list[str] | None = None) -> str:
        name = original_fix(name, extra_prefixes)
        for prefix, new_prefix in hf_to_vllm_prefix.items():
            if name.startswith(prefix):
                return name.replace(prefix, new_prefix, 1)
        return name

    vllm_generation._fix_param_name_to_vllm = _fix_param_name_to_vllm
    vllm_generation._brickgpt_qwen35_prefix_patch = True
    print(f"Patched VLLMGeneration._fix_param_name_to_vllm for {model_architectures}")
    logger.info("Patched VLLMGeneration._fix_param_name_to_vllm for %s", model_architectures)


def patch_grpo_deepspeed_model_device_mismatch(trainer) -> None:
    """Redirect TRL's CPU-side `self.model` calls to the active DeepSpeed-wrapped model."""

    if getattr(trainer, "_brickgpt_deepspeed_model_patch", False):
        return

    original_get = getattr(trainer, "_get_per_token_logps_and_entropies", None)
    if original_get is None:
        return

    def _get_per_token_logps_and_entropies(model, *args, **kwargs):
        active_model = model
        wrapped_model = getattr(trainer, "model_wrapped", None)
        if (
            getattr(trainer, "is_deepspeed_enabled", False)
            and model is getattr(trainer, "model", None)
            and wrapped_model is not None
            and wrapped_model is not model
        ):
            active_model = wrapped_model
            if not getattr(trainer, "_brickgpt_reported_deepspeed_model_redirect", False):
                print("Redirecting GRPO per-token logps from self.model to DeepSpeed model_wrapped")
                trainer._brickgpt_reported_deepspeed_model_redirect = True
        return original_get(active_model, *args, **kwargs)

    trainer._get_per_token_logps_and_entropies = _get_per_token_logps_and_entropies
    trainer._brickgpt_deepspeed_model_patch = True
    logger.info("Patched GRPOTrainer._get_per_token_logps_and_entropies for DeepSpeed model_wrapped")
