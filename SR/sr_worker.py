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
ROI_REQUEST_MAGIC = 0x32525341
RECT = struct.Struct("<4I")
FRAME_FLAG_ROI = 0x4
FRAME_FLAG_BACKGROUND = 0x8
FRAME_FLAG_BUDGET = 0x10


def prepare_regions(regions, width, height, margin=12):
    """Clip/pad and merge intersecting rectangles until no overlap remains."""
    boxes = []
    for x, y, w, h in regions:
        if not w or not h or x >= width or y >= height:
            raise ValueError("invalid ROI bounds")
        box = (max(0, x - margin), max(0, y - margin),
               min(width, x + w + margin), min(height, y + h + margin))
        index = 0
        while index < len(boxes):
            other = boxes[index]
            if (box[0] <= other[2] and other[0] <= box[2]
                    and box[1] <= other[3] and other[1] <= box[3]):
                box = (min(box[0], other[0]), min(box[1], other[1]),
                       max(box[2], other[2]), max(box[3], other[3]))
                boxes.pop(index)
                index = 0
            else:
                index += 1
        boxes.append(box)
    return boxes


def selective_upscale(rgb, scale, regions, infer, fallback="bicubic"):
    """Experimental ROI SR; infer=None means no model is available."""
    height, width = rgb.shape[:2]
    boxes = prepare_regions(regions, width, height)
    if infer is None:
        return fallback_upscale(rgb, scale, fallback), FRAME_FLAG_FALLBACK
    if len(boxes) > 8 or sum((r-l)*(b-t) for l,t,r,b in boxes) >= width*height*0.6:
        return infer(rgb), FRAME_FLAG_MODEL | FRAME_FLAG_BUDGET
    output = fallback_upscale(rgb, scale, fallback)
    if not boxes:
        return output, FRAME_FLAG_BACKGROUND
    for left, top, right, bottom in boxes:
        enhanced = infer(rgb[top:bottom, left:right].copy())
        target = output[top*scale:bottom*scale, left*scale:right*scale]
        # Feather only the outer padded edge, leaving the object interior at
        # full model strength. This reduces seams, not temporal flicker.
        h, w = target.shape[:2]
        feather = 4 * scale
        xs, ys = np.arange(w), np.arange(h)
        wx = np.minimum(xs + 1, w - xs) / feather
        wy = np.minimum(ys + 1, h - ys) / feather
        if left == 0: wx[:feather] = 1
        if right == width: wx[-feather:] = 1
        if top == 0: wy[:feather] = 1
        if bottom == height: wy[-feather:] = 1
        alpha = np.minimum(1, np.minimum(wy[:, None], wx[None, :]))[..., None]
        target[:] = np.rint(enhanced * alpha + target * (1 - alpha)).astype(np.uint8)
    return output, FRAME_FLAG_MODEL | FRAME_FLAG_ROI


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
    # Opt-in recording preserves the binary stdout protocol and original RGB.
    recorder = None
    if os.environ.get('AVION_ROI_RECORD_DIR'):
        from roi_dataset import RoiRecorder
        recorder = RoiRecorder.from_environment()

    while True:
        header_bytes = read_exact(stdin, HEADER.size)

        if header_bytes is None:
            return 0

        magic, width, height, pitch, pixel_format, _reserved, payload_size = HEADER.unpack(
            header_bytes
        )

        if magic not in (REQUEST_MAGIC, ROI_REQUEST_MAGIC):
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

        if payload_size > 64 * 1024 * 1024 or _reserved > 64:
            raise ValueError("frame/ROI limit exceeded")
        regions = []
        if magic == ROI_REQUEST_MAGIC:
            raw_regions = read_exact(stdin, _reserved * RECT.size)
            if raw_regions is None:
                return 0
            regions = list(RECT.iter_unpack(raw_regions))
        payload = read_exact(stdin, payload_size)

        if payload is None:
            return 0

        rgb = argb8888_bytes_to_rgb(payload, width, height, pitch)

        if recorder is not None and magic == ROI_REQUEST_MAGIC:
            try:
                recorder.record(rgb, regions)
            except Exception as exc:
                log(f'ROI recording stopped: {exc}')
                recorder = None

        if magic == ROI_REQUEST_MAGIC:
            # Crop sizes vary each frame: avoid cuDNN autotuning every shape.
            if torch is not None:
                torch.backends.cudnn.benchmark = False
            infer = (lambda crop: model_upscale(torch, model, device, crop)) if model is not None else None
            output_rgb, output_flags = selective_upscale(rgb, args.scale, regions, infer, args.fallback)
        elif model is not None:
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
