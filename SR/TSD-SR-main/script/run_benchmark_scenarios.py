import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


SCENARIOS_QUICK = [
    {"name": "baseline_wavelet_512", "args": {"align_method": "wavelet", "process_size": 512}},
    {"name": "adain_512", "args": {"align_method": "adain", "process_size": 512}},
    {"name": "nofix_512", "args": {"align_method": "nofix", "process_size": 512}},
    {
        "name": "tile_wavelet_512",
        "args": {
            "align_method": "wavelet",
            "process_size": 512,
            "is_use_tile": True,
            "latent_tiled_size": 64,
            "latent_tiled_overlap": 8,
        },
    },
    {"name": "wavelet_384", "args": {"align_method": "wavelet", "process_size": 384}},
    {"name": "wavelet_768", "args": {"align_method": "wavelet", "process_size": 768}},
]

SCENARIOS_EXTENDED = SCENARIOS_QUICK + [
    {
        "name": "tile_wavelet_768_overlap16",
        "args": {
            "align_method": "wavelet",
            "process_size": 768,
            "is_use_tile": True,
            "latent_tiled_size": 96,
            "latent_tiled_overlap": 16,
        },
    },
    {"name": "adain_384", "args": {"align_method": "adain", "process_size": 384}},
    {"name": "adain_768", "args": {"align_method": "adain", "process_size": 768}},
    {"name": "wavelet_fp32", "args": {"align_method": "wavelet", "process_size": 512, "mixed_precision": "fp32"}},
]

PERTURBATIONS_DEFAULT = [
    "clean",
    "gaussian_noise_10",
    "gaussian_noise_25",
    "jpeg_30",
    "jpeg_10",
    "blur_3",
    "blur_7",
    "low_light_06",
]

METRIC_KEYS = ["PSNR", "SSIM", "LPIPS", "DISTS", "CLIPIQA", "NIQE", "MUSIQ", "MANIQA", "FID"]


@dataclass
class DatasetSpec:
    name: str
    lr_dir: Path
    gt_dir: Optional[Path]


def parse_dataset_spec(spec: str) -> DatasetSpec:
    parts = [p.strip() for p in spec.split("|")]
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid --dataset format: {spec}. Use name|lr_dir|gt_dir or name|lr_dir")
    name = parts[0]
    lr_dir = Path(parts[1])
    gt_dir = Path(parts[2]) if len(parts) == 3 and parts[2] else None
    if not lr_dir.exists():
        raise FileNotFoundError(f"LR directory not found: {lr_dir}")
    if gt_dir is not None and not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    return DatasetSpec(name=name, lr_dir=lr_dir, gt_dir=gt_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TSD-SR across multiple scenarios and perturbations.")
    parser.add_argument("--model_family", choices=["auto", "sd3", "sdxl"], default="auto")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="checkpoint/tsdsr")
    parser.add_argument("--lora_dir", type=str, default="checkpoint/tsdsr")
    parser.add_argument("--embedding_dir", type=str, default="dataset/default")
    parser.add_argument(
        "--dataset",
        action="append",
        default=["Test|imgs/test"],
        help="Dataset spec format: name|lr_dir|gt_dir (gt_dir optional). Repeat this flag for multiple datasets.",
    )
    parser.add_argument("--scenario_set", choices=["quick", "extended"], default="quick")
    parser.add_argument("--scenario_file", type=str, default=None, help="Optional JSON file overriding scenario definitions.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--output_root", type=str, default="outputs/benchmarks")
    parser.add_argument("--log_root", type=str, default="logs/benchmarks")
    parser.add_argument("--temp_root", type=str, default="outputs/benchmarks_tmp")
    parser.add_argument("--enable_perturbations", action="store_true")
    parser.add_argument(
        "--perturbations",
        nargs="+",
        default=PERTURBATIONS_DEFAULT,
        help="Perturbations: clean, gaussian_noise_10, gaussian_noise_25, jpeg_30, jpeg_10, blur_3, blur_7, low_light_06",
    )
    parser.add_argument("--max_runs", type=int, default=None, help="Optional safety cap for total runs.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_scenarios(args: argparse.Namespace) -> List[Dict]:
    if args.scenario_file:
        with open(args.scenario_file, "r", encoding="utf-8") as f:
            scenarios = json.load(f)
        if not isinstance(scenarios, list):
            raise ValueError("Scenario file must be a list of objects with keys: name, args")
        return scenarios
    return SCENARIOS_QUICK if args.scenario_set == "quick" else SCENARIOS_EXTENDED


def apply_perturbation(img_bgr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "clean":
        return img_bgr
    if mode.startswith("gaussian_noise_"):
        sigma = float(mode.split("_")[-1])
        noise = np.random.normal(0, sigma, img_bgr.shape).astype(np.float32)
        out = np.clip(img_bgr.astype(np.float32) + noise, 0, 255)
        return out.astype(np.uint8)
    if mode.startswith("jpeg_"):
        q = int(mode.split("_")[-1])
        ok, encoded = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("JPEG decoding failed")
        return decoded
    if mode.startswith("blur_"):
        k = int(mode.split("_")[-1])
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(img_bgr, (k, k), 0)
    if mode == "low_light_06":
        return np.clip(img_bgr.astype(np.float32) * 0.6, 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported perturbation: {mode}")


def build_perturbation_dirs(dataset: DatasetSpec, perturbations: List[str], temp_root: Path) -> Dict[str, Path]:
    image_paths = sorted(dataset.lr_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No PNG files found in {dataset.lr_dir}")

    result: Dict[str, Path] = {}
    for mode in perturbations:
        if mode == "clean":
            result[mode] = dataset.lr_dir
            continue

        out_dir = temp_root / dataset.name / mode
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in image_paths:
            dst = out_dir / img_path.name
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image: {img_path}")
            out = apply_perturbation(img, mode)
            cv2.imwrite(str(dst), out)

        result[mode] = out_dir

    return result


def run_command(cmd: List[str], dry_run: bool) -> None:
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return
    env = os.environ.copy()
    cwd = str(Path.cwd())
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = f"{cwd}{os.pathsep}{current_pythonpath}"
    else:
        env["PYTHONPATH"] = cwd
    subprocess.run(cmd, check=True, env=env)


def find_new_log(log_dir: Path, before: set) -> Optional[Path]:
    after = set(log_dir.glob("*.log"))
    diff = sorted(after - before, key=lambda p: p.stat().st_mtime)
    return diff[-1] if diff else None


def parse_metrics_from_log(log_path: Path) -> Dict[str, float]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.findall(r"Average Metrics for .*?\n(.*?FID:\s*[0-9.]+)", text, flags=re.DOTALL)
    target = blocks[-1] if blocks else text
    metrics: Dict[str, float] = {}
    for key in METRIC_KEYS:
        m = re.search(rf"\b{re.escape(key)}:\s*([0-9]+(?:\.[0-9]+)?)", target)
        if m:
            metrics[key] = float(m.group(1))
    return metrics


def main() -> None:
    args = parse_args()
    datasets = [parse_dataset_spec(spec) for spec in args.dataset]
    scenarios = load_scenarios(args)

    output_root = Path(args.output_root)
    log_root = Path(args.log_root)
    temp_root = Path(args.temp_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    if args.enable_perturbations:
        temp_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    total_runs = 0

    for dataset in datasets:
        if args.enable_perturbations:
            pert_map = build_perturbation_dirs(dataset, args.perturbations, temp_root)
        else:
            pert_map = {"clean": dataset.lr_dir}

        for pert_name, lr_dir in pert_map.items():
            for scenario in scenarios:
                total_runs += 1
                if args.max_runs is not None and total_runs > args.max_runs:
                    print(f"[INFO] Reached --max_runs={args.max_runs}. Stopping.")
                    break

                scenario_name = scenario["name"]
                run_id = f"{dataset.name}__{pert_name}__{scenario_name}"
                out_dir = output_root / run_id
                run_log_dir = log_root / run_id
                run_log_dir.mkdir(parents=True, exist_ok=True)

                row = {
                    "run_id": run_id,
                    "dataset": dataset.name,
                    "perturbation": pert_name,
                    "scenario": scenario_name,
                    "input_dir": str(lr_dir),
                    "gt_dir": str(dataset.gt_dir) if dataset.gt_dir else "",
                    "output_dir": str(out_dir),
                    "status": "pending",
                    "infer_seconds": "",
                    "metric_log": "",
                }

                if out_dir.exists() and args.overwrite:
                    shutil.rmtree(out_dir)

                infer_cmd = [
                    args.python,
                    "test/test_tsdsr.py",
                    "--model_family",
                    args.model_family,
                    "--pretrained_model_name_or_path",
                    args.pretrained_model_name_or_path,
                    "--lora_dir",
                    args.lora_dir,
                    "--embedding_dir",
                    args.embedding_dir,
                    "--device",
                    args.device,
                    "--upscale",
                    str(args.upscale),
                    "-i",
                    str(lr_dir),
                    "-o",
                    str(out_dir),
                ]

                for k, v in scenario.get("args", {}).items():
                    key = f"--{k}"
                    if isinstance(v, bool):
                        if v:
                            infer_cmd.extend([key, "True"])
                    else:
                        infer_cmd.extend([key, str(v)])

                start = time.time()
                try:
                    run_command(infer_cmd, args.dry_run)
                    row["infer_seconds"] = round(time.time() - start, 3)
                    row["status"] = "inference_ok"
                except subprocess.CalledProcessError as e:
                    row["status"] = f"inference_failed({e.returncode})"
                    all_rows.append(row)
                    continue

                if dataset.gt_dir is None:
                    all_rows.append(row)
                    continue

                before_logs = set(run_log_dir.glob("*.log"))
                metric_cmd = [
                    args.python,
                    "test/test_metrics.py",
                    "--inp_imgs",
                    str(out_dir),
                    "--gt_imgs",
                    str(dataset.gt_dir),
                    "--log",
                    str(run_log_dir),
                ]
                try:
                    run_command(metric_cmd, args.dry_run)
                    row["status"] = "metrics_ok"
                except subprocess.CalledProcessError as e:
                    row["status"] = f"metrics_failed({e.returncode})"
                    all_rows.append(row)
                    continue

                if not args.dry_run:
                    new_log = find_new_log(run_log_dir, before_logs)
                    if new_log:
                        row["metric_log"] = str(new_log)
                        metrics = parse_metrics_from_log(new_log)
                        row.update({k: metrics.get(k, "") for k in METRIC_KEYS})

                all_rows.append(row)

            if args.max_runs is not None and total_runs >= args.max_runs:
                break
        if args.max_runs is not None and total_runs >= args.max_runs:
            break

    summary_json = log_root / "benchmark_summary.json"
    summary_csv = log_root / "benchmark_summary.csv"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "run_id",
        "dataset",
        "perturbation",
        "scenario",
        "input_dir",
        "gt_dir",
        "output_dir",
        "status",
        "infer_seconds",
        "metric_log",
    ] + METRIC_KEYS

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"[DONE] Saved summary JSON: {summary_json}")
    print(f"[DONE] Saved summary CSV : {summary_csv}")


if __name__ == "__main__":
    main()
