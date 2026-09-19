#!/usr/bin/env python3
import argparse
import os
import struct
import sys
from typing import Optional

import numpy as np

sys.dont_write_bytecode = True


REQUEST_MAGIC = 0x31525341
RESPONSE_MAGIC = 0x31525352
FORMAT_ARGB8888 = 1
FRAME_FLAG_MODEL = 0x1
FRAME_FLAG_FALLBACK = 0x2
HEADER = struct.Struct("<7I")


def log(message: str) -> None:
    print(f"SR worker: {message}", file=sys.stderr, flush=True)


def read_exact(stream, size: int) -> Optional[bytes]:
    chunks = []
    remaining = size

    while remaining > 0:
        chunk = stream.read(remaining)

        if not chunk:
            return None

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def default_checkpoint(scale: int) -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, "IMDN", "checkpoints", f"IMDN_x{scale}.pth")


def load_imdn(checkpoint: str, scale: int, device_name: str, strict_model: bool):
    try:
        import torch
    except Exception as exc:
        if strict_model:
            raise

        log(f"torch import failed; using fallback upscale ({exc})")
        return None, None, None

    sr_root = os.path.dirname(os.path.abspath(__file__))
    imdn_root = os.path.join(sr_root, "IMDN")

    if imdn_root not in sys.path:
        sys.path.insert(0, imdn_root)

    try:
        import utils
        from model import architecture

        if device_name == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_name)

        model = architecture.IMDN(upscale=scale)
        model.load_state_dict(utils.load_state_dict(checkpoint), strict=True)
        model.eval().to(device)

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        log(f"loaded IMDN x{scale} on {device}: {checkpoint}")
        return torch, model, device
    except Exception as exc:
        if strict_model:
            raise

        log(f"model load failed; using fallback upscale ({exc})")
        return None, None, None


def fallback_upscale(rgb: np.ndarray, scale: int, mode: str) -> np.ndarray:
    if mode == "nearest":
        return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)

    try:
        import cv2

        interpolation = {
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
        }.get(mode, cv2.INTER_CUBIC)

        return cv2.resize(
            rgb,
            (rgb.shape[1] * scale, rgb.shape[0] * scale),
            interpolation=interpolation,
        )
    except Exception:
        return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def model_upscale(torch, model, device, rgb: np.ndarray) -> np.ndarray:
    input_tensor = torch.from_numpy(rgb).to(device=device)
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

    with torch.no_grad():
        output = model(input_tensor).clamp_(0.0, 1.0)

    output = output.squeeze(0).permute(1, 2, 0).mul_(255.0).byte()
    return output.cpu().numpy()


def argb8888_bytes_to_rgb(payload: bytes, width: int, height: int, pitch: int) -> np.ndarray:
    rows = np.frombuffer(payload, dtype=np.uint8).reshape(height, pitch)
    bgra = rows[:, : width * 4].reshape(height, width, 4)
    return bgra[:, :, [2, 1, 0]].copy()


def rgb_to_argb8888_bytes(rgb: np.ndarray) -> tuple[bytes, int, int, int]:
    height, width, _ = rgb.shape
    bgra = np.empty((height, width, 4), dtype=np.uint8)
    bgra[:, :, 0] = rgb[:, :, 2]
    bgra[:, :, 1] = rgb[:, :, 1]
    bgra[:, :, 2] = rgb[:, :, 0]
    bgra[:, :, 3] = 255

    pitch = width * 4
    return bgra.tobytes(), width, height, pitch


def process_frames(args) -> int:
    checkpoint = args.checkpoint or default_checkpoint(args.scale)
    torch, model, device = load_imdn(
        checkpoint,
        args.scale,
        args.device,
        args.strict_model,
    )

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        header_bytes = read_exact(stdin, HEADER.size)

        if header_bytes is None:
            return 0

        magic, width, height, pitch, pixel_format, _reserved, payload_size = HEADER.unpack(
            header_bytes
        )

        if magic != REQUEST_MAGIC:
            log(f"invalid request magic: 0x{magic:08x}")
            return 2

        if pixel_format != FORMAT_ARGB8888:
            log(f"unsupported pixel format: {pixel_format}")
            return 2

        expected_size = pitch * height

        if width <= 0 or height <= 0 or pitch < width * 4 or payload_size != expected_size:
            log(
                "invalid frame header: "
                f"width={width}, height={height}, pitch={pitch}, bytes={payload_size}"
            )
            return 2

        payload = read_exact(stdin, payload_size)

        if payload is None:
            return 0

        rgb = argb8888_bytes_to_rgb(payload, width, height, pitch)

        if model is not None:
            output_rgb = model_upscale(torch, model, device, rgb)
            output_flags = FRAME_FLAG_MODEL
        else:
            output_rgb = fallback_upscale(rgb, args.scale, args.fallback)
            output_flags = FRAME_FLAG_FALLBACK

        output_payload, output_width, output_height, output_pitch = rgb_to_argb8888_bytes(
            output_rgb
        )

        stdout.write(
            HEADER.pack(
                RESPONSE_MAGIC,
                output_width,
                output_height,
                output_pitch,
                FORMAT_ARGB8888,
                output_flags,
                len(output_payload),
            )
        )
        stdout.write(output_payload)
        stdout.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Avion live SR worker")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--fallback",
        choices=["nearest", "bilinear", "bicubic"],
        default="bicubic",
    )
    parser.add_argument("--strict-model", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(process_frames(parse_args()))
    except Exception as exc:
        log(f"fatal error: {exc}")
        raise
