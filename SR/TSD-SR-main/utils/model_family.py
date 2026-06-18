import json
from pathlib import Path
from typing import Tuple

import torch
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)


DEFAULT_PROMPT = (
    "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo "
    "- realistic maximum detail, 32k, Color Grading, ultra HD, extrememeticulous detailing, skin pore "
    "detailing, hyper sharpness, perfect without deformations."
)


def resolve_model_family(pretrained_model_name_or_path: str, requested_family: str = "auto") -> str:
    if requested_family != "auto":
        return requested_family

    model_path = Path(pretrained_model_name_or_path)
    if model_path.exists():
        if (model_path / "transformer").exists():
            return "sd3"
        if (model_path / "unet").exists():
            return "sdxl"
        model_index_path = model_path / "model_index.json"
        if model_index_path.exists():
            model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
            class_name = str(model_index.get("_class_name", "")).lower()
            if "stablediffusion3" in class_name:
                return "sd3"
            if "stablediffusionxl" in class_name:
                return "sdxl"

    lower_name = pretrained_model_name_or_path.lower()
    if "juggernaut" in lower_name or "sdxl" in lower_name:
        return "sdxl"
    return "sd3"


def _encode_prompt_sd3(prompt: str, pretrained_model_name_or_path: str, device: str, dtype: torch.dtype):
    tokenizer_one = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
    tokenizer_three = T5TokenizerFast.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_3")

    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    ).to(device)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_2"
    ).to(device)
    text_encoder_three = T5EncoderModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_3"
    ).to(device)

    prompt_list = [prompt]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in (
        (tokenizer_one, text_encoder_one),
        (tokenizer_two, text_encoder_two),
    ):
        text_inputs = tokenizer(
            prompt_list,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        prompt_outputs = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
        clip_pooled_prompt_embeds_list.append(prompt_outputs[0])
        clip_prompt_embeds_list.append(prompt_outputs.hidden_states[-2])

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    text_inputs = tokenizer_three(
        prompt_list,
        padding="max_length",
        max_length=256,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    t5_prompt_embed = text_encoder_three(text_inputs.input_ids.to(device))[0]

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds.to(dtype=dtype), pooled_prompt_embeds.to(dtype=dtype)


def _encode_prompt_sdxl(prompt: str, pretrained_model_name_or_path: str, device: str, dtype: torch.dtype):
    tokenizer_one = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")

    text_encoder_one = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_2"
    ).to(device)

    prompt_list = [prompt]
    prompt_embeds_list = []
    pooled_prompt_embeds = None

    for tokenizer, text_encoder in (
        (tokenizer_one, text_encoder_one),
        (tokenizer_two, text_encoder_two),
    ):
        text_inputs = tokenizer(
            prompt_list,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_outputs = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
        if pooled_prompt_embeds is None and prompt_outputs[0].ndim == 2:
            pooled_prompt_embeds = prompt_outputs[0]
        prompt_embeds_list.append(prompt_outputs.hidden_states[-2])

    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
    return prompt_embeds.to(dtype=dtype), pooled_prompt_embeds.to(dtype=dtype)


def encode_prompt(prompt: str, pretrained_model_name_or_path: str, model_family: str, device: str, dtype: torch.dtype):
    if model_family == "sd3":
        return _encode_prompt_sd3(prompt, pretrained_model_name_or_path, device, dtype)
    if model_family == "sdxl":
        return _encode_prompt_sdxl(prompt, pretrained_model_name_or_path, device, dtype)
    raise ValueError(f"Unsupported model family: {model_family}")


def load_prompt_embeddings(
    embedding_dir: str,
    pretrained_model_name_or_path: str,
    model_family: str,
    device: str,
    dtype: torch.dtype,
    prompt: str = DEFAULT_PROMPT,
):
    if embedding_dir:
        prompt_path = Path(embedding_dir) / "prompt_embeds.pt"
        pool_path = Path(embedding_dir) / "pool_embeds.pt"
        if prompt_path.exists() and pool_path.exists():
            prompt_embeds = torch.load(prompt_path, map_location=device, weights_only=False).to(dtype=dtype)
            pooled_prompt_embeds = torch.load(pool_path, map_location=device, weights_only=False).to(dtype=dtype)
            return prompt_embeds, pooled_prompt_embeds

    return encode_prompt(prompt, pretrained_model_name_or_path, model_family, device, dtype)


def get_sdxl_text_projection_dim(pretrained_model_name_or_path: str) -> int:
    text_encoder_2_config = CLIPTextConfig.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_2")
    return text_encoder_2_config.projection_dim


def build_sdxl_add_time_ids(
    original_size: Tuple[int, int],
    target_size: Tuple[int, int],
    projection_dim: int,
    unet,
    device: str,
    dtype: torch.dtype,
):
    add_time_ids = list(original_size + (0, 0) + target_size)
    passed_add_embed_dim = unet.config.addition_time_embed_dim * len(add_time_ids) + projection_dim
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features
    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Unexpected SDXL add_time_ids dim: expected {expected_add_embed_dim}, got {passed_add_embed_dim}."
        )
    return torch.tensor([add_time_ids], device=device, dtype=dtype)
