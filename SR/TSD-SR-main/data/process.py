import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import AutoencoderKL

from utils.model_family import DEFAULT_PROMPT, resolve_model_family


DIV2K_PATH = os.path.join(os.path.dirname(__file__), "DIV2K", "train")

data_path = [DIV2K_PATH]
default_model_path = "/path/to/your/sd3_model"
hr_dir_name = "gt"
prompt_dir_name = "prompt_txt"
prompt_embeds_dir_name = "prompt_embeds"
pool_prompt_embeds_dir_name = "pool_embeds"
hr_latnet_dir_name = "latent_hr"
lr_dir_name = "lr_bicubic"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _sort_key(path_or_name):
    name = Path(path_or_name).name
    stem = Path(name).stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def _list_images(path):
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file: {path}")
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"{path} not exist")
    return sorted(
        [file_path for file_path in path.rglob("*") if file_path.suffix.lower() in IMAGE_EXTENSIONS],
        key=_sort_key,
    )


def merge_data(root_dirs, hr_name=hr_dir_name):
    hr_data_file_path = []
    if isinstance(root_dirs, (str, os.PathLike)):
        root_dirs = [root_dirs]

    for data_dir in root_dirs:
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"{data_dir} not exist")

        hr_data_dir = os.path.join(data_dir, hr_name)
        hr_file = [
            file_name
            for file_name in os.listdir(hr_data_dir)
            if os.path.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS
        ]
        hr_file.sort(key=_sort_key)
        hr_data_file_path.extend(os.path.join(hr_data_dir, file_name) for file_name in hr_file)
    return hr_data_file_path


def prepare_training_images(
    raw_image_dir,
    output_dir,
    process_size=512,
    downscale_factor=4,
    prompt_text=DEFAULT_PROMPT,
    overwrite=False,
):
    raw_images = _list_images(raw_image_dir)
    if not raw_images:
        raise FileNotFoundError(f"No supported images found in {raw_image_dir}")

    output_dir = Path(output_dir)
    gt_dir = output_dir / hr_dir_name
    lr_dir = output_dir / lr_dir_name
    prompt_dir = output_dir / prompt_dir_name
    for directory in (gt_dir, lr_dir, prompt_dir):
        directory.mkdir(parents=True, exist_ok=True)

    lr_size = max(1, process_size // downscale_factor)
    name_width = max(4, len(str(len(raw_images))))

    for index, image_path in enumerate(raw_images, 1):
        stem = f"{index:0{name_width}d}"
        gt_path = gt_dir / f"{stem}.png"
        lr_path = lr_dir / f"{stem}.png"
        prompt_path = prompt_dir / f"{stem}.txt"

        if not overwrite and gt_path.exists() and lr_path.exists() and prompt_path.exists():
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            crop_size = min(width, height, process_size)
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            hr_image = image.crop((left, top, left + crop_size, top + crop_size))
            if crop_size != process_size:
                hr_image = hr_image.resize((process_size, process_size), Image.Resampling.LANCZOS)

            lr_image = hr_image.resize((lr_size, lr_size), Image.Resampling.BICUBIC)
            lr_image = lr_image.resize((process_size, process_size), Image.Resampling.BICUBIC)

            hr_image.save(gt_path)
            lr_image.save(lr_path)

        prompt_path.write_text(prompt_text.rstrip() + "\n", encoding="utf-8")

    return str(output_dir)


def _encode_prompt_with_t5(text_encoder, tokenizer, prompt=None, num_images_per_prompt=1, device=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=256,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds


def _encode_prompt_with_clip(text_encoder, tokenizer, prompt, device=None, num_images_per_prompt=1, max_length=77):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    prompt_outputs = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = prompt_outputs[0]
    prompt_embeds = prompt_outputs.hidden_states[-2].to(dtype=text_encoder.dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds, pooled_prompt_embeds


def encode_prompt_sd3(text_encoders, tokenizers, prompt, device=None, num_images_per_prompt=1):
    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(tokenizers[:2], text_encoders[:2]):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            max_length=77,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )
    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    return prompt_embeds, pooled_prompt_embeds


def encode_prompt_sdxl(text_encoders, tokenizers, prompt, device=None, num_images_per_prompt=1):
    prompt_embeds_list = []
    pooled_prompt_embeds = None
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        prompt_outputs = text_encoder(
            tokenizer(
                [prompt] if isinstance(prompt, str) else prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device if device is not None else text_encoder.device),
            output_hidden_states=True,
        )
        if pooled_prompt_embeds is None and prompt_outputs[0].ndim == 2:
            pooled_prompt_embeds = prompt_outputs[0]
        prompt_embeds = prompt_outputs.hidden_states[-2].to(
            dtype=text_encoder.dtype,
            device=device if device is not None else text_encoder.device,
        )
        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
    return prompt_embeds, pooled_prompt_embeds


def import_text_encoder_sd3(pretrained_model_name_or_path=default_model_path, device=None):
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
    return [tokenizer_one, tokenizer_two, tokenizer_three], [text_encoder_one, text_encoder_two, text_encoder_three]


def import_text_encoder_sdxl(pretrained_model_name_or_path=default_model_path, device=None):
    tokenizer_one = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_one = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_2"
    ).to(device)
    return [tokenizer_one, tokenizer_two], [text_encoder_one, text_encoder_two]


def import_text_encoders(pretrained_model_name_or_path=default_model_path, device=None, model_family="sd3"):
    if model_family == "sd3":
        return import_text_encoder_sd3(pretrained_model_name_or_path, device=device)
    if model_family == "sdxl":
        return import_text_encoder_sdxl(pretrained_model_name_or_path, device=device)
    raise ValueError(f"Unsupported model family: {model_family}")


def encode_prompt_for_family(text_encoders, tokenizers, prompt, model_family, device=None):
    if model_family == "sd3":
        return encode_prompt_sd3(text_encoders, tokenizers, prompt, device=device)
    if model_family == "sdxl":
        return encode_prompt_sdxl(text_encoders, tokenizers, prompt, device=device)
    raise ValueError(f"Unsupported model family: {model_family}")


def run_encode_prompt(root_dirs, pretrained_model_name_or_path=default_model_path, device="cuda", model_family="auto"):
    model_family = resolve_model_family(pretrained_model_name_or_path, model_family)
    if isinstance(root_dirs, (str, os.PathLike)):
        root_dirs = [root_dirs]
    hr_data_file = merge_data(root_dirs)
    for data_dir in root_dirs:
        os.makedirs(os.path.join(data_dir, prompt_embeds_dir_name), exist_ok=True)
        os.makedirs(os.path.join(data_dir, pool_prompt_embeds_dir_name), exist_ok=True)

    hr_data_file = [
        hr_img_file
        for hr_img_file in hr_data_file
        if not (
            os.path.exists(
                os.path.join(
                    os.path.dirname(os.path.dirname(hr_img_file)),
                    prompt_embeds_dir_name,
                    f"{os.path.splitext(os.path.basename(hr_img_file))[0]}.pt",
                )
            )
            and os.path.exists(
                os.path.join(
                    os.path.dirname(os.path.dirname(hr_img_file)),
                    pool_prompt_embeds_dir_name,
                    f"{os.path.splitext(os.path.basename(hr_img_file))[0]}.pt",
                )
            )
        )
    ]
    if not hr_data_file:
        print("Prompt embedding cache is already complete.")
        return

    tokenizers, text_encoders = import_text_encoders(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        device=device,
        model_family=model_family,
    )

    for hr_img_file in tqdm(hr_data_file, total=len(hr_data_file)):
        stem = os.path.splitext(os.path.basename(hr_img_file))[0]
        data_dir = os.path.dirname(os.path.dirname(hr_img_file))
        prompt_file = os.path.join(data_dir, prompt_dir_name, f"{stem}.txt")
        prompt_path = os.path.join(data_dir, prompt_embeds_dir_name, f"{stem}.pt")
        pool_path = os.path.join(data_dir, pool_prompt_embeds_dir_name, f"{stem}.pt")
        if os.path.exists(prompt_path) and os.path.exists(pool_path):
            continue

        with open(prompt_file, "r", encoding="utf-8") as file_obj:
            prompt = file_obj.read()

        prompt_embeds, pooled_prompt_embeds = encode_prompt_for_family(
            text_encoders,
            tokenizers,
            prompt,
            model_family=model_family,
            device=device,
        )
        torch.save(prompt_embeds.detach().cpu(), prompt_path)
        torch.save(pooled_prompt_embeds.detach().cpu(), pool_path)
        print(f"{prompt_path} Done !")
        del prompt_embeds, pooled_prompt_embeds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_single_prompt_embeddings(
    output_embedding_dir,
    prompt_text,
    pretrained_model_name_or_path=default_model_path,
    device="cuda",
    model_family="auto",
):
    model_family = resolve_model_family(pretrained_model_name_or_path, model_family)
    os.makedirs(output_embedding_dir, exist_ok=True)

    tokenizers, text_encoders = import_text_encoders(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        device=device,
        model_family=model_family,
    )
    prompt_embeds, pooled_prompt_embeds = encode_prompt_for_family(
        text_encoders,
        tokenizers,
        prompt_text,
        model_family=model_family,
        device=device,
    )
    torch.save(prompt_embeds.detach().cpu(), os.path.join(output_embedding_dir, "prompt_embeds.pt"))
    torch.save(pooled_prompt_embeds.detach().cpu(), os.path.join(output_embedding_dir, "pool_embeds.pt"))
    print(f"Saved prompt embeddings to {output_embedding_dir}")


def vae_encode(hr_img_paths, hr_latent_path=hr_latnet_dir_name, model_path=default_model_path, device="cuda", weight_dtype=torch.float32):
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(device, weight_dtype)
    trans = transforms.ToTensor()
    for hr_img_file in tqdm(hr_img_paths, total=len(hr_img_paths)):
        stem = os.path.splitext(os.path.basename(hr_img_file))[0]
        data_dir = os.path.dirname(os.path.dirname(hr_img_file))
        hr_save_path = os.path.join(data_dir, hr_latent_path, f"{stem}.pt")
        if os.path.exists(hr_save_path):
            continue
        hr_img = Image.open(hr_img_file).convert("RGB")
        hq = trans(hr_img).unsqueeze(0).to(device, dtype=weight_dtype) * 2 - 1
        hq_latent = vae.encode(hq).latent_dist.sample() * vae.config.scaling_factor
        torch.save(hq_latent.detach().cpu(), hr_save_path)
        print(f"{os.path.basename(hr_save_path)} Done !")
        del hq, hq_latent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_vae_encode(root_dirs, pretrained_model_name_or_path=default_model_path, device="cuda"):
    if isinstance(root_dirs, (str, os.PathLike)):
        root_dirs = [root_dirs]
    hr_data_file = merge_data(root_dirs)
    for data_dir in root_dirs:
        os.makedirs(os.path.join(data_dir, hr_latnet_dir_name), exist_ok=True)
    hr_data_file = [
        hr_img_file
        for hr_img_file in hr_data_file
        if not os.path.exists(
            os.path.join(
                os.path.dirname(os.path.dirname(hr_img_file)),
                hr_latnet_dir_name,
                f"{os.path.splitext(os.path.basename(hr_img_file))[0]}.pt",
            )
        )
    ]
    if not hr_data_file:
        print("HR latent cache is already complete.")
        return
    vae_encode(hr_data_file, model_path=pretrained_model_name_or_path, device=device)


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute prompt embeddings and HR latents for TSD-SR training.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=default_model_path,
        help="Base model used to encode prompt embeddings and HR latents.",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        default="auto",
        choices=["auto", "sd3", "sdxl"],
        help="Model family for prompt encoding and inference cache generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device used for preprocessing.",
    )
    parser.add_argument(
        "--skip_latents",
        action="store_true",
        help="Skip VAE latent preprocessing.",
    )
    parser.add_argument(
        "--skip_prompts",
        action="store_true",
        help="Skip prompt embedding preprocessing.",
    )
    parser.add_argument(
        "--raw_image_dir",
        type=str,
        default=None,
        help="Optional image file or directory to convert into a prepared training dataset.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=DIV2K_PATH,
        help="Prepared training dataset directory.",
    )
    parser.add_argument(
        "--process_size",
        type=int,
        default=512,
        help="Square crop/resize size used when preparing raw images.",
    )
    parser.add_argument(
        "--downscale_factor",
        type=int,
        default=4,
        help="Bicubic degradation downscale factor used when preparing raw images.",
    )
    parser.add_argument(
        "--overwrite_prepared_data",
        action="store_true",
        help="Overwrite existing prepared gt/lr/prompt files.",
    )
    parser.add_argument(
        "--output_embedding_dir",
        type=str,
        default=None,
        help="Optional directory to save a single prompt embedding bundle.",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt text used with --output_embedding_dir.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dirs = [args.train_data_dir]

    if args.raw_image_dir is not None:
        root_dirs = [
            prepare_training_images(
                args.raw_image_dir,
                args.train_data_dir,
                process_size=args.process_size,
                downscale_factor=args.downscale_factor,
                prompt_text=args.prompt_text,
                overwrite=args.overwrite_prepared_data,
            )
        ]

    if args.output_embedding_dir is not None:
        save_single_prompt_embeddings(
            args.output_embedding_dir,
            args.prompt_text,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            device=args.device,
            model_family=args.model_family,
        )

    if not args.skip_latents:
        run_vae_encode(root_dirs, pretrained_model_name_or_path=args.pretrained_model_name_or_path, device=args.device)
    if not args.skip_prompts:
        run_encode_prompt(
            root_dirs,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            device=args.device,
            model_family=args.model_family,
        )
    print("All done !")
