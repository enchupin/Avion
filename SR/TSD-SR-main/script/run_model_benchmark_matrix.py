import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ModelSpec:
    name: str
    model_family: str
    pretrained_model_name_or_path: str
    lora_dir: str
    embedding_dir: str


def parse_model_spec(spec: str) -> ModelSpec:
    parts = [part.strip() for part in spec.split("|")]
    if len(parts) not in (4, 5):
        raise ValueError(
            "Invalid --model format. Use: name|pretrained_model_name_or_path|lora_dir|embedding_dir or "
            "name|model_family|pretrained_model_name_or_path|lora_dir|embedding_dir"
        )

    if len(parts) == 4:
        name, pretrained_model_name_or_path, lora_dir, embedding_dir = parts
        model_family = "auto"
    else:
        name, model_family, pretrained_model_name_or_path, lora_dir, embedding_dir = parts

    return ModelSpec(
        name=name,
        model_family=model_family,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        lora_dir=lora_dir,
        embedding_dir=embedding_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark scenarios for multiple model configs and aggregate the results."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "Format: name|pretrained_model_name_or_path|lora_dir|embedding_dir or "
            "name|model_family|pretrained_model_name_or_path|lora_dir|embedding_dir. Repeat for multiple models."
        ),
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--output_root", type=str, default="outputs/model_benchmarks")
    parser.add_argument("--log_root", type=str, default="logs/model_benchmarks")
    parser.add_argument(
        "--benchmark_script",
        type=str,
        default="script/run_benchmark_scenarios.py",
        help="Path to the per-model benchmark runner.",
    )
    return parser.parse_known_args()


def run_command(cmd: List[str], cwd: Path) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> None:
    args, forwarded_args = parse_args()
    root = Path(__file__).resolve().parents[1]
    benchmark_script = (root / args.benchmark_script).resolve()
    output_root = (root / args.output_root).resolve()
    log_root = (root / args.log_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    model_specs = [parse_model_spec(spec) for spec in args.model]
    combined_rows = []

    for model_spec in model_specs:
        model_output_root = output_root / model_spec.name
        model_log_root = log_root / model_spec.name
        model_output_root.mkdir(parents=True, exist_ok=True)
        model_log_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python,
            str(benchmark_script),
            "--model_family",
            model_spec.model_family,
            "--pretrained_model_name_or_path",
            model_spec.pretrained_model_name_or_path,
            "--lora_dir",
            model_spec.lora_dir,
            "--embedding_dir",
            model_spec.embedding_dir,
            "--output_root",
            str(model_output_root),
            "--log_root",
            str(model_log_root),
        ] + forwarded_args
        run_command(cmd, root)

        summary_json = model_log_root / "benchmark_summary.json"
        with open(summary_json, "r", encoding="utf-8") as file_obj:
            model_rows = json.load(file_obj)

        for row in model_rows:
            row["model_name"] = model_spec.name
            row["model_family"] = model_spec.model_family
            row["model_pretrained_model_name_or_path"] = model_spec.pretrained_model_name_or_path
            row["model_lora_dir"] = model_spec.lora_dir
            row["model_embedding_dir"] = model_spec.embedding_dir
            combined_rows.append(row)

    combined_json = log_root / "model_matrix_summary.json"
    combined_csv = log_root / "model_matrix_summary.csv"

    with open(combined_json, "w", encoding="utf-8") as file_obj:
        json.dump(combined_rows, file_obj, ensure_ascii=False, indent=2)

    fieldnames = [
        "model_name",
        "model_family",
        "model_pretrained_model_name_or_path",
        "model_lora_dir",
        "model_embedding_dir",
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
        "PSNR",
        "SSIM",
        "LPIPS",
        "DISTS",
        "CLIPIQA",
        "NIQE",
        "MUSIQ",
        "MANIQA",
        "FID",
    ]

    with open(combined_csv, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in combined_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"[DONE] Saved combined JSON: {combined_json}")
    print(f"[DONE] Saved combined CSV : {combined_csv}")


if __name__ == "__main__":
    main()
