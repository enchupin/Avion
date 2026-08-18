import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    aliases: tuple[str, ...]
    requires_gt: bool
    direction: str
    kwargs: dict[str, Any]
    description: str


METRIC_SPECS = {
    "psnr_y": MetricSpec(
        name="psnr_y",
        aliases=("psnr",),
        requires_gt=True,
        direction="higher",
        kwargs={"test_y_channel": True, "color_space": "ycbcr"},
        description="PSNR on the Y channel of YCbCr, as reported for fidelity.",
    ),
    "ssim_y": MetricSpec(
        name="ssim_y",
        aliases=("ssim",),
        requires_gt=True,
        direction="higher",
        kwargs={"test_y_channel": True, "color_space": "ycbcr"},
        description="SSIM on the Y channel of YCbCr, as reported for fidelity.",
    ),
    "lpips": MetricSpec(
        name="lpips",
        aliases=("lpips",),
        requires_gt=True,
        direction="lower",
        kwargs={},
        description="LPIPS full-reference perceptual distance.",
    ),
    "dists": MetricSpec(
        name="dists",
        aliases=("dists",),
        requires_gt=True,
        direction="lower",
        kwargs={},
        description="DISTS full-reference perceptual distance.",
    ),
    "fid": MetricSpec(
        name="fid",
        aliases=("fid",),
        requires_gt=True,
        direction="lower",
        kwargs={},
        description="FID distribution distance between generated and GT image sets.",
    ),
    "niqe": MetricSpec(
        name="niqe",
        aliases=("niqe",),
        requires_gt=False,
        direction="lower",
        kwargs={},
        description="NIQE no-reference natural image quality score.",
    ),
    "musiq": MetricSpec(
        name="musiq",
        aliases=("musiq",),
        requires_gt=False,
        direction="higher",
        kwargs={},
        description="MUSIQ no-reference image quality score.",
    ),
    "maniqa": MetricSpec(
        name="maniqa",
        aliases=("maniqa-pipal", "maniqa"),
        requires_gt=False,
        direction="higher",
        kwargs={},
        description="MANIQA no-reference image quality score.",
    ),
    "clipiqa": MetricSpec(
        name="clipiqa",
        aliases=("clipiqa",),
        requires_gt=False,
        direction="higher",
        kwargs={},
        description="CLIPIQA no-reference image quality score.",
    ),
    "topiq": MetricSpec(
        name="topiq",
        aliases=("topiq_nr", "topiq_iaa"),
        requires_gt=False,
        direction="higher",
        kwargs={},
        description="TOPIQ no-reference score for estimating perceptual image quality.",
    ),
    "qalign": MetricSpec(
        name="qalign",
        aliases=("qalign",),
        requires_gt=False,
        direction="higher",
        kwargs={},
        description="Q-Align no-reference score for estimating human-aligned image quality.",
    ),
}


PRESET_METRICS = {
    "tsdsr": (
        "psnr_y",
        "ssim_y",
        "lpips",
        "dists",
        "fid",
        "niqe",
        "musiq",
        "maniqa",
        "clipiqa",
    ),
    "tinysr": (
        "psnr_y",
        "ssim_y",
        "lpips",
        "dists",
        "fid",
        "niqe",
        "musiq",
        "maniqa",
        "clipiqa",
        "topiq",
        "qalign",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SR output folders using the metrics reported by the "
            "TSD-SR and TinySR papers. This script only evaluates already "
            "generated images; it does not run a model."
        )
    )
    parser.add_argument("--pred", nargs="+", required=True, help="SR/result image directories.")
    parser.add_argument("--gt", nargs="*", default=None, help="GT image directories. Required for full-reference metrics.")
    parser.add_argument("--dataset-name", nargs="*", default=None, help="Optional names for each --pred directory.")
    parser.add_argument("--preset", choices=("tsdsr", "tinysr", "both"), default="both")
    parser.add_argument("--metrics", nargs="*", default=None, help="Explicit metric names. Overrides --preset.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--output", default="logs/paper_metrics", help="Directory for CSV/JSON reports.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N matched images.")
    parser.add_argument("--strict", action="store_true", help="Fail if a requested pyiqa metric is unavailable.")
    parser.add_argument("--skip-fid", action="store_true", help="Skip FID even if the selected preset includes it.")
    parser.add_argument("--resize-pred-to-gt", action="store_true", help="Resize prediction to GT size before pair metrics.")
    parser.add_argument("--efficiency-json", default=None, help="Optional JSON with steps/time/MACs/params to copy into summary.")
    return parser.parse_args()


def resolve_metrics(args: argparse.Namespace) -> list[str]:
    if args.metrics:
        metrics = args.metrics
    elif args.preset == "both":
        metrics = sorted(set(PRESET_METRICS["tsdsr"]) | set(PRESET_METRICS["tinysr"]))
    else:
        metrics = list(PRESET_METRICS[args.preset])

    normalized = []
    for metric in metrics:
        key = metric.lower().replace("-", "_")
        if key == "psnr":
            key = "psnr_y"
        if key == "ssim":
            key = "ssim_y"
        if key not in METRIC_SPECS:
            raise ValueError(f"Unknown metric: {metric}. Available: {', '.join(sorted(METRIC_SPECS))}")
        if key == "fid" and args.skip_fid:
            continue
        if key not in normalized:
            normalized.append(key)
    return normalized


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def image_key(path: Path) -> str:
    return path.stem


def pair_images(pred_dir: Path, gt_dir: Path | None, limit: int | None) -> list[tuple[Path, Path | None]]:
    pred_paths = list_images(pred_dir)
    if gt_dir is None:
        pairs = [(path, None) for path in pred_paths]
        return pairs[:limit] if limit else pairs

    gt_by_key = {image_key(path): path for path in list_images(gt_dir)}
    pairs = []
    missing = []
    for pred_path in pred_paths:
        gt_path = gt_by_key.get(image_key(pred_path))
        if gt_path is None:
            missing.append(pred_path.name)
            continue
        pairs.append((pred_path, gt_path))

    if missing:
        print(f"[WARN] {pred_dir}: {len(missing)} prediction images had no matching GT by stem.", file=sys.stderr)
    return pairs[:limit] if limit else pairs


def load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device)


def resize_like(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if pred.shape[-2:] == gt.shape[-2:]:
        return pred
    return torch.nn.functional.interpolate(pred, size=gt.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)


def create_metric(spec: MetricSpec, device: torch.device, strict: bool):
    try:
        import pyiqa
    except ImportError as exc:
        raise RuntimeError("pyiqa is required for paper metric evaluation. Install it before running this script.") from exc

    last_error = None
    for alias in spec.aliases:
        try:
            metric = pyiqa.create_metric(alias, device=device, **spec.kwargs)
            return alias, metric.to(device) if hasattr(metric, "to") else metric
        except Exception as exc:
            last_error = exc

    message = f"Unable to create metric {spec.name} with aliases {spec.aliases}: {last_error}"
    if strict:
        raise RuntimeError(message)
    print(f"[WARN] {message}", file=sys.stderr)
    return None, None


def mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def compute_directory(
    dataset_name: str,
    pred_dir: Path,
    gt_dir: Path | None,
    metric_names: list[str],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    pairs = pair_images(pred_dir, gt_dir, args.limit)
    if not pairs:
        raise RuntimeError(f"No images to evaluate for {dataset_name}: {pred_dir}")

    metric_objects = {}
    unavailable = []
    for name in metric_names:
        if name == "fid":
            continue
        spec = METRIC_SPECS[name]
        if spec.requires_gt and gt_dir is None:
            unavailable.append({"metric": name, "reason": "GT directory was not provided."})
            continue
        alias, metric = create_metric(spec, device, args.strict)
        if metric is None:
            unavailable.append({"metric": name, "reason": "pyiqa metric unavailable."})
            continue
        metric_objects[name] = {"alias": alias, "metric": metric}

    per_image_rows = []
    accum = {name: [] for name in metric_objects}

    for index, (pred_path, gt_path) in enumerate(pairs, start=1):
        pred = load_rgb_tensor(pred_path, device)
        gt = load_rgb_tensor(gt_path, device) if gt_path else None
        if gt is not None and args.resize_pred_to_gt:
            pred = resize_like(pred, gt)

        row = {
            "dataset": dataset_name,
            "index": index,
            "pred": str(pred_path),
            "gt": str(gt_path) if gt_path else "",
        }

        with torch.no_grad():
            for name, payload in metric_objects.items():
                spec = METRIC_SPECS[name]
                metric = payload["metric"]
                if spec.requires_gt:
                    if gt is None:
                        continue
                    if pred.shape != gt.shape:
                        raise ValueError(
                            f"Shape mismatch for {pred_path.name}: pred={tuple(pred.shape)}, gt={tuple(gt.shape)}. "
                            "Use --resize-pred-to-gt if this is expected."
                        )
                    value = metric(pred, gt)
                else:
                    value = metric(pred)

                value_float = float(value.detach().cpu().item() if torch.is_tensor(value) else value)
                row[name] = value_float
                accum[name].append(value_float)

        per_image_rows.append(row)
        print(f"[{dataset_name}] {index}/{len(pairs)} {pred_path.name}")

    summary = {
        "dataset": dataset_name,
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir) if gt_dir else None,
        "num_images": len(pairs),
        "metrics": {},
        "unavailable_metrics": unavailable,
    }

    for name, values in accum.items():
        summary["metrics"][name] = {
            "mean": mean(values),
            "direction": METRIC_SPECS[name].direction,
            "pyiqa_name": metric_objects[name]["alias"],
            "description": METRIC_SPECS[name].description,
        }

    if "fid" in metric_names:
        if gt_dir is None:
            summary["unavailable_metrics"].append({"metric": "fid", "reason": "GT directory was not provided."})
        else:
            alias, fid_metric = create_metric(METRIC_SPECS["fid"], device, args.strict)
            if fid_metric is None:
                summary["unavailable_metrics"].append({"metric": "fid", "reason": "pyiqa metric unavailable."})
            else:
                with torch.no_grad():
                    fid_value = fid_metric(str(gt_dir), str(pred_dir))
                summary["metrics"]["fid"] = {
                    "mean": float(fid_value.detach().cpu().item() if torch.is_tensor(fid_value) else fid_value),
                    "direction": METRIC_SPECS["fid"].direction,
                    "pyiqa_name": alias,
                    "description": METRIC_SPECS["fid"].description,
                }

    return {"summary": summary, "per_image": per_image_rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_efficiency_json(path: str | None) -> Any:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    metric_names = resolve_metrics(args)
    pred_dirs = [Path(path).resolve() for path in args.pred]
    gt_dirs = [Path(path).resolve() for path in args.gt] if args.gt else []

    if gt_dirs and len(gt_dirs) != len(pred_dirs):
        raise ValueError("--gt must be omitted or have the same number of entries as --pred.")

    if args.dataset_name and len(args.dataset_name) != len(pred_dirs):
        raise ValueError("--dataset-name must have the same number of entries as --pred.")

    device = get_device(args.device)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_results = []
    all_rows = []

    for index, pred_dir in enumerate(pred_dirs):
        dataset_name = args.dataset_name[index] if args.dataset_name else pred_dir.name
        gt_dir = gt_dirs[index] if gt_dirs else None
        result = compute_directory(dataset_name, pred_dir, gt_dir, metric_names, device, args)
        dataset_results.append(result["summary"])
        all_rows.extend(result["per_image"])

    report = {
        "created_at": run_id,
        "preset": args.preset,
        "requested_metrics": metric_names,
        "device": str(device),
        "paper_metric_notes": {
            "TSD-SR": list(PRESET_METRICS["tsdsr"]),
            "TinySR": list(PRESET_METRICS["tinysr"]),
            "efficiency": "TinySR additionally reports #Steps, inference time, MACs, and parameters. Provide them through --efficiency-json if measured elsewhere.",
        },
        "efficiency": load_efficiency_json(args.efficiency_json),
        "datasets": dataset_results,
    }

    summary_path = output_dir / f"paper_metrics_summary_{run_id}.json"
    rows_path = output_dir / f"paper_metrics_per_image_{run_id}.csv"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(rows_path, all_rows)

    print(f"Summary: {summary_path}")
    print(f"Per-image CSV: {rows_path}")


if __name__ == "__main__":
    main()
