import argparse
import os
import subprocess
import sys
from pathlib import Path


PAPER_DATASETS = {
    "DRealSR": ("DrealSRVal_crop128/test_LR", "DrealSRVal_crop128/test_HR"),
    "RealSR": ("RealSRVal_crop128/test_LR", "RealSRVal_crop128/test_HR"),
    "DIV2K-Val": ("DIV2K_V2_val/test_LR", "DIV2K_V2_val/test_HR"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the TSD-SR paper evaluation settings.")
    parser.add_argument("--model_family", choices=["auto", "sd3", "sdxl"], default="auto")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default="imgs/StableSR_testsets")
    parser.add_argument("--datasets", nargs="+", choices=sorted(PAPER_DATASETS.keys()), default=list(PAPER_DATASETS.keys()))
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--lora_dir", type=str, default="checkpoint/tsdsr-mse")
    parser.add_argument("--embedding_dir", type=str, default="dataset/default")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_root", type=str, default="outputs/paper_eval")
    parser.add_argument("--log_root", type=str, default="logs/paper_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def run_command(cmd, root):
    print("[CMD]", " ".join(cmd))
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = f"{root}{os.pathsep}{current_pythonpath}"
    else:
        env["PYTHONPATH"] = str(root)
    subprocess.run(cmd, check=True, cwd=root, env=env)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset_root = (root / args.dataset_root).resolve()
    output_root = (root / args.output_root).resolve()
    log_root = (root / args.log_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        lr_rel, gt_rel = PAPER_DATASETS[dataset_name]
        lr_dir = (dataset_root / lr_rel).resolve()
        gt_dir = (dataset_root / gt_rel).resolve()

        if not lr_dir.exists():
            raise FileNotFoundError(f"Missing LR directory for {dataset_name}: {lr_dir}")
        if not gt_dir.exists():
            raise FileNotFoundError(f"Missing GT directory for {dataset_name}: {gt_dir}")

        output_dir = output_root / dataset_name
        metric_log_dir = log_root / dataset_name
        metric_log_dir.mkdir(parents=True, exist_ok=True)

        inference_cmd = [
            args.python,
            "test/test_tsdsr.py",
            "--paper_eval",
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
            "--seed",
            str(args.seed),
            "-i",
            str(lr_dir),
            "-o",
            str(output_dir),
        ]
        if args.deterministic:
            inference_cmd.append("--deterministic")

        metric_cmd = [
            args.python,
            "test/test_metrics.py",
            "--inp_imgs",
            str(output_dir),
            "--gt_imgs",
            str(gt_dir),
            "--log",
            str(metric_log_dir),
        ]

        run_command(inference_cmd, root)
        run_command(metric_cmd, root)


if __name__ == "__main__":
    main()
