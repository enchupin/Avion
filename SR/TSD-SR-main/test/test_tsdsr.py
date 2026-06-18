import argparse
import glob
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.append(".")

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig
from torchvision import transforms
from tqdm import tqdm
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline, StableDiffusionXLPipeline, UNet2DConditionModel

from models.autoencoder_kl import AutoencoderKL
from utils.model_family import (
    DEFAULT_PROMPT,
    build_sdxl_add_time_ids,
    get_sdxl_text_projection_dim,
    load_prompt_embeddings,
    resolve_model_family,
)
from utils.util import load_lora_state_dict
from utils.vaehook import _init_tiled_vae
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix


DEFAULT_LORA_DIR = "checkpoint/tsdsr"
PAPER_EVAL_LORA_DIR = "checkpoint/tsdsr-mse"
DEFAULT_ALIGN_METHOD = "wavelet"
PAPER_EVAL_ALIGN_METHOD = "adain"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
SDXL_UNET_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]
VAE_TARGET_MODULES = [
    "encoder.conv_in",
    "encoder.down_blocks.0.resnets.0.conv1",
    "encoder.down_blocks.0.resnets.0.conv2",
    "encoder.down_blocks.0.resnets.1.conv1",
    "encoder.down_blocks.0.resnets.1.conv2",
    "encoder.down_blocks.0.downsamplers.0.conv",
    "encoder.down_blocks.1.resnets.0.conv1",
    "encoder.down_blocks.1.resnets.0.conv2",
    "encoder.down_blocks.1.resnets.0.conv_shortcut",
    "encoder.down_blocks.1.resnets.1.conv1",
    "encoder.down_blocks.1.resnets.1.conv2",
    "encoder.down_blocks.1.downsamplers.0.conv",
    "encoder.down_blocks.2.resnets.0.conv1",
    "encoder.down_blocks.2.resnets.0.conv2",
    "encoder.down_blocks.2.resnets.0.conv_shortcut",
    "encoder.down_blocks.2.resnets.1.conv1",
    "encoder.down_blocks.2.resnets.1.conv2",
    "encoder.down_blocks.2.downsamplers.0.conv",
    "encoder.down_blocks.3.resnets.0.conv1",
    "encoder.down_blocks.3.resnets.0.conv2",
    "encoder.down_blocks.3.resnets.1.conv1",
    "encoder.down_blocks.3.resnets.1.conv2",
    "encoder.mid_block.attentions.0.to_q",
    "encoder.mid_block.attentions.0.to_k",
    "encoder.mid_block.attentions.0.to_v",
    "encoder.mid_block.attentions.0.to_out.0",
    "encoder.mid_block.resnets.0.conv1",
    "encoder.mid_block.resnets.0.conv2",
    "encoder.mid_block.resnets.1.conv1",
    "encoder.mid_block.resnets.1.conv2",
    "encoder.conv_out",
    "quant_conv",
]


@dataclass
class RuntimeState:
    model_family: str
    denoiser: torch.nn.Module
    vae: torch.nn.Module
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    timesteps: torch.Tensor
    weight_dtype: torch.dtype
    sdxl_projection_dim: int | None = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_family", type=str, default="auto", choices=["auto", "sd3", "sdxl"])
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="path/to/your/sd3",
        required=True,
        help="Path to the pretrained base model.",
    )
    parser.add_argument("--lora_dir", type=str, default=DEFAULT_LORA_DIR, help="path to tsd-sr lora weights")
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/", help="path to prompt embeddings")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs/", help="path to save results")
    parser.add_argument("--input_dir", "-i", type=str, default="path/to/your/image/folder", required=True)
    parser.add_argument("--default_prompt", type=str, default=DEFAULT_PROMPT)

    parser.add_argument("--rank", type=int, default=64, help="rank for denoiser lora")
    parser.add_argument("--rank_vae", type=int, default=64, help="rank for vae")

    parser.add_argument("--is_use_tile", type=bool, default=False, help="whether to use tiled vae")
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224, help="tiled size for tiled vae decoder")
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024, help="tiled size for tiled vae encoder")
    parser.add_argument("--latent_tiled_size", type=int, default=64, help="tiled size for transformer latent")
    parser.add_argument("--latent_tiled_overlap", type=int, default=8, help="tiled overlap for transformer latent")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", help="enable deterministic inference settings")
    parser.add_argument(
        "--paper_eval",
        action="store_true",
        help="apply the paper evaluation profile (checkpoint/tsdsr-mse + adain) unless explicitly overridden",
    )
    parser.add_argument("--upscale", type=int, default=4, help="upscale factor")
    parser.add_argument("--process_size", type=int, default=512, help="process size for images")
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument(
        "--align_method",
        type=str,
        choices=["wavelet", "adain", "nofix"],
        default="wavelet",
        help="color alignment method",
    )
    parser.add_argument("--lora_transformer_weight_name", type=str, default="transformer.safetensors")
    parser.add_argument("--lora_unet_weight_name", type=str, default="unet.safetensors")
    parser.add_argument("--lora_vae_weight_name", type=str, default="vae.safetensors")

    args = parser.parse_args()
    if args.paper_eval:
        if args.lora_dir == DEFAULT_LORA_DIR:
            args.lora_dir = PAPER_EVAL_LORA_DIR
        if args.align_method == DEFAULT_ALIGN_METHOD:
            args.align_method = PAPER_EVAL_ALIGN_METHOD
    return args


def set_reproducibility(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def build_latent_generator(args):
    generator_device = "cpu"
    if args.device.startswith("cuda") and torch.cuda.is_available():
        generator_device = args.device

    generator = torch.Generator(device=generator_device)
    generator.manual_seed(args.seed)
    return generator


def _gaussian_weights(tile_width, tile_height, nbatches, channels, device):
    from numpy import exp, outer, pi, sqrt

    var = 0.01
    midpoint_x = (tile_width - 1) / 2
    x_probs = [
        exp(-(x - midpoint_x) * (x - midpoint_x) / (tile_width * tile_width) / (2 * var)) / sqrt(2 * pi * var)
        for x in range(tile_width)
    ]
    midpoint_y = tile_height / 2
    y_probs = [
        exp(-(y - midpoint_y) * (y - midpoint_y) / (tile_height * tile_height) / (2 * var)) / sqrt(2 * pi * var)
        for y in range(tile_height)
    ]
    weights = outer(y_probs, x_probs)
    return torch.tile(torch.tensor(weights, device=device), (nbatches, channels, 1, 1))


def _resolve_lora_weight_name(lora_dir, candidates):
    lora_path = Path(lora_dir)
    if lora_path.is_file():
        return lora_path.name

    for candidate in candidates:
        if candidate and (lora_path / candidate).exists():
            return candidate
    return None


def _ensure_batch_prompt_embeds(prompt_embeds, batch_size, device, dtype):
    if prompt_embeds.ndim == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
    if prompt_embeds.shape[0] == 1 and batch_size > 1:
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
    return prompt_embeds.to(device=device, dtype=dtype)


def _ensure_batch_pooled_embeds(pooled_prompt_embeds, batch_size, device, dtype):
    if pooled_prompt_embeds.ndim == 1:
        pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0)
    if pooled_prompt_embeds.shape[0] == 1 and batch_size > 1:
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(batch_size, 1)
    return pooled_prompt_embeds.to(device=device, dtype=dtype)


def _prepare_runtime_embeddings(args, model_family, weight_dtype):
    prompt_embeds, pooled_prompt_embeds = load_prompt_embeddings(
        args.embedding_dir,
        args.pretrained_model_name_or_path,
        model_family,
        args.device,
        weight_dtype,
        prompt=args.default_prompt,
    )
    return prompt_embeds, pooled_prompt_embeds


def _build_sd3_runtime(args, weight_dtype):
    denoiser = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype)
    if args.is_use_tile:
        _init_tiled_vae(vae, encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "proj", "linear", "proj_out"],
    )
    denoiser.add_adapter(transformer_lora_config)
    denoiser.enable_adapters()

    vae_lora_config = LoraConfig(
        r=args.rank_vae,
        lora_alpha=args.rank_vae,
        init_lora_weights="gaussian",
        target_modules=VAE_TARGET_MODULES,
    )
    vae.add_adapter(vae_lora_config)
    vae.enable_adapters()

    transformer_weight_name = _resolve_lora_weight_name(
        args.lora_dir,
        [args.lora_transformer_weight_name, "transformer.safetensors"],
    )
    if transformer_weight_name is not None:
        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(
            args.lora_dir, weight_name=transformer_weight_name
        )
        load_lora_state_dict(transformer_lora_state_dict, denoiser, module_prefixes=("transformer",))

    vae_weight_name = _resolve_lora_weight_name(args.lora_dir, [args.lora_vae_weight_name, "vae.safetensors"])
    if vae_weight_name is not None:
        vae_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.lora_dir, weight_name=vae_weight_name)
        load_lora_state_dict(vae_lora_state_dict, vae, module_prefixes=("transformer", "vae"))

    prompt_embeds, pooled_prompt_embeds = _prepare_runtime_embeddings(args, "sd3", weight_dtype)
    return RuntimeState(
        model_family="sd3",
        denoiser=denoiser.to(args.device, dtype=weight_dtype).eval(),
        vae=vae.to(args.device, dtype=weight_dtype).eval(),
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        timesteps=torch.tensor([1000.0], device=args.device, dtype=weight_dtype),
        weight_dtype=weight_dtype,
    )


def _build_sdxl_runtime(args, weight_dtype):
    denoiser = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype)
    if args.is_use_tile:
        _init_tiled_vae(vae, encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=SDXL_UNET_TARGET_MODULES,
    )
    denoiser.add_adapter(unet_lora_config)
    denoiser.enable_adapters()

    vae_lora_config = LoraConfig(
        r=args.rank_vae,
        lora_alpha=args.rank_vae,
        init_lora_weights="gaussian",
        target_modules=VAE_TARGET_MODULES,
    )
    vae.add_adapter(vae_lora_config)
    vae.enable_adapters()

    unet_weight_name = _resolve_lora_weight_name(
        args.lora_dir,
        [args.lora_unet_weight_name, "transformer.safetensors", "pytorch_lora_weights.safetensors"],
    )
    if unet_weight_name is not None:
        unet_lora_state_dict = StableDiffusionXLPipeline.lora_state_dict(args.lora_dir, weight_name=unet_weight_name)
        load_lora_state_dict(unet_lora_state_dict, denoiser, module_prefixes=("unet", "transformer"))

    vae_weight_name = _resolve_lora_weight_name(args.lora_dir, [args.lora_vae_weight_name, "vae.safetensors"])
    if vae_weight_name is not None:
        vae_lora_state_dict = StableDiffusionXLPipeline.lora_state_dict(args.lora_dir, weight_name=vae_weight_name)
        load_lora_state_dict(vae_lora_state_dict, vae, module_prefixes=("vae", "transformer"))

    prompt_embeds, pooled_prompt_embeds = _prepare_runtime_embeddings(args, "sdxl", weight_dtype)
    projection_dim = get_sdxl_text_projection_dim(args.pretrained_model_name_or_path)
    return RuntimeState(
        model_family="sdxl",
        denoiser=denoiser.to(args.device, dtype=weight_dtype).eval(),
        vae=vae.to(args.device, dtype=weight_dtype).eval(),
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        timesteps=torch.tensor([999.0], device=args.device, dtype=weight_dtype),
        weight_dtype=weight_dtype,
        sdxl_projection_dim=projection_dim,
    )


def build_runtime(args, weight_dtype):
    model_family = resolve_model_family(args.pretrained_model_name_or_path, args.model_family)
    if model_family == "sd3":
        return _build_sd3_runtime(args, weight_dtype)
    if model_family == "sdxl":
        return _build_sdxl_runtime(args, weight_dtype)
    raise ValueError(f"Unsupported model family: {model_family}")


def predict_denoiser(runtime, sample, timesteps, original_size, target_size, args):
    batch_size = sample.shape[0]
    batch_prompt_embeds = _ensure_batch_prompt_embeds(
        runtime.prompt_embeds, batch_size, args.device, runtime.weight_dtype
    )
    batch_pooled_prompt_embeds = _ensure_batch_pooled_embeds(
        runtime.pooled_prompt_embeds, batch_size, args.device, runtime.weight_dtype
    )
    if timesteps.ndim == 1 and timesteps.shape[0] == 1 and batch_size > 1:
        timesteps = timesteps.repeat(batch_size)

    if runtime.model_family == "sd3":
        return runtime.denoiser(
            hidden_states=sample,
            timestep=timesteps,
            encoder_hidden_states=batch_prompt_embeds,
            pooled_projections=batch_pooled_prompt_embeds,
            return_dict=False,
        )[0]

    add_time_ids = build_sdxl_add_time_ids(
        original_size,
        target_size,
        runtime.sdxl_projection_dim,
        runtime.denoiser,
        args.device,
        runtime.weight_dtype,
    ).repeat(batch_size, 1)
    return runtime.denoiser(
        sample=sample,
        timestep=timesteps,
        encoder_hidden_states=batch_prompt_embeds,
        added_cond_kwargs={
            "text_embeds": batch_pooled_prompt_embeds,
            "time_ids": add_time_ids,
        },
        return_dict=False,
    )[0]


def tile_sample(lq_latent, lq, runtime, args, original_size, target_size):
    with torch.no_grad():
        _, channels, h, w = lq_latent.size()
        tile_size, tile_overlap = args.latent_tiled_size, args.latent_tiled_overlap
        if h * w <= tile_size * tile_size:
            return predict_denoiser(runtime, lq_latent, runtime.timesteps, original_size, target_size, args).to(
                args.device, dtype=runtime.weight_dtype
            )

        print(f"[Tiled Latent]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")
        tile_size = min(tile_size, min(h, w))
        tile_weights = _gaussian_weights(tile_size, tile_size, 1, channels, args.device)

        grid_rows = 0
        cur_x = 0
        while cur_x < lq_latent.size(-1):
            cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
            grid_rows += 1

        grid_cols = 0
        cur_y = 0
        while cur_y < lq_latent.size(-2):
            cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
            grid_cols += 1

        input_list = []
        noise_preds = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols - 1 or row < grid_rows - 1:
                    ofs_x = max(row * tile_size - tile_overlap * row, 0)
                    ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1:
                    ofs_x = w - tile_size
                if col == grid_cols - 1:
                    ofs_y = h - tile_size

                input_tile = lq_latent[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size]
                input_list.append(input_tile)

                if len(input_list) == 1 or col == grid_cols - 1:
                    input_tiles = torch.cat(input_list, dim=0).to(args.device, dtype=runtime.weight_dtype)
                    model_out = predict_denoiser(runtime, input_tiles, runtime.timesteps, original_size, target_size, args)
                    input_list = []
                noise_preds.append(model_out)

        noise_pred = torch.zeros(lq_latent.shape, device=args.device)
        contributors = torch.zeros(lq_latent.shape, device=args.device)
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols - 1 or row < grid_rows - 1:
                    ofs_x = max(row * tile_size - tile_overlap * row, 0)
                    ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1:
                    ofs_x = w - tile_size
                if col == grid_cols - 1:
                    ofs_y = h - tile_size

                noise_pred[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size] += (
                    noise_preds[row * grid_cols + col] * tile_weights
                )
                contributors[:, :, ofs_y : ofs_y + tile_size, ofs_x : ofs_x + tile_size] += tile_weights

        noise_pred /= contributors
        return noise_pred.to(args.device, dtype=runtime.weight_dtype)


tensor_transforms = transforms.Compose([transforms.ToTensor()])


def run_model(args, pixel_values, size, latent_generator, runtime):
    with torch.no_grad():
        pixel_values = torch.nn.functional.interpolate(pixel_values, size=size, mode="bicubic", align_corners=False)
        pixel_values = pixel_values * 2 - 1
        pixel_values = pixel_values.to(args.device, dtype=runtime.weight_dtype).clamp(-1, 1)

        model_input = runtime.vae.encode(pixel_values).latent_dist.sample(generator=latent_generator)
        model_input = model_input * runtime.vae.config.scaling_factor
        model_input = model_input.to(args.device, dtype=runtime.weight_dtype)

        model_pred = tile_sample(model_input, pixel_values, runtime, args, size, size)
        latent_stu = model_input - model_pred
        image = runtime.vae.decode(latent_stu / runtime.vae.config.scaling_factor, return_dict=False)[0]
        return image.squeeze(0).clamp(-1, 1)


if __name__ == "__main__":
    args = parse_args()
    set_reproducibility(args)
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    runtime = build_runtime(args, weight_dtype)
    latent_generator = build_latent_generator(args)

    if os.path.isdir(args.input_dir):
        image_names = sorted(
            path
            for path in glob.glob(os.path.join(args.input_dir, "*"))
            if os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
        )
    else:
        image_names = [args.input_dir]

    datalen = len(image_names)
    print("image_num", datalen)
    os.makedirs(args.output_dir, exist_ok=True)

    total_time = 0.0
    for image_name in tqdm(image_names):
        lr = Image.open(image_name).convert("RGB")
        ori_width, ori_height = lr.size
        upscale = args.upscale
        process_size = args.process_size

        resize_flag = False
        if ori_width < process_size // upscale or ori_height < process_size // upscale:
            scale = (process_size // upscale) / min(ori_width, ori_height)
            new_width, new_height = int(scale * ori_width), int(scale * ori_height)
            resize_flag = True
        else:
            new_width, new_height = ori_width, ori_height
        new_width, new_height = upscale * new_width, upscale * new_height
        if new_width % 8 or new_height % 8:
            resize_flag = True
            new_width = new_width - new_width % 8
            new_height = new_height - new_height % 8

        lr_scale = lr.resize((int(ori_width * upscale), int(ori_height * upscale)))
        pixel_values = tensor_transforms(lr).unsqueeze(0).to(args.device, dtype=runtime.weight_dtype)
        start_time = time.time()
        image = run_model(args, pixel_values, (new_height, new_width), latent_generator, runtime)
        end_time = time.time()
        image_pil_image = transforms.ToPILImage()(image.cpu() / 2 + 0.5)
        total_time += end_time - start_time

        if resize_flag:
            image_pil_image = image_pil_image.resize((int(ori_width * upscale), int(ori_height * upscale)))

        if args.align_method == "adain":
            image_pil_image = adain_color_fix(target=image_pil_image, source=lr)
        elif args.align_method == "wavelet":
            image_pil_image = wavelet_color_fix(target=image_pil_image, source=lr_scale)

        image_pil_image.save(os.path.join(args.output_dir, os.path.basename(image_name)))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Average time: {total_time / datalen}")
