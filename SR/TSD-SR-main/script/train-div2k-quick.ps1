$ErrorActionPreference = "Stop"

$env:MODEL_NAME = "checkpoint/tsdsr"
$env:DEFAULT_EMBED = "dataset/default"
$env:NULL_EMBED = "dataset/null"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:WANDB_MODE = "offline"
$env:OUTPUT_DIR = "checkpoint/tsdsr-save-div2k-quick/"
$env:OUTPUT_LOG = "logs/tsdsr_div2k_quick.log"
$env:LOG_NAME = "tsdsr-div2k-quick"

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$ErrorActionPreference = "Continue"
accelerate launch --num_processes 1 --mixed_precision="fp16" train/train.py `
  --pretrained_model_name_or_path=$env:MODEL_NAME `
  --train_data_dir data/DIV2K/train_quick_256_64 `
  --default_embedding_dir=$env:DEFAULT_EMBED --null_embedding_dir=$env:NULL_EMBED `
  --train_batch_size=1 --rank=16 --rank_vae=16 --rank_lora=16 `
  --num_train_epochs=1 --checkpointing_steps=10 --validation_steps=1000 --max_train_steps=30 `
  --learning_rate=5e-06 --learning_rate_reg=1e-06 --lr_scheduler="cosine_with_restarts" --lr_warmup_steps=3 `
  --seed=43 --use_default_prompt --use_random_bias --gradient_checkpointing `
  --output_dir=$env:OUTPUT_DIR `
  --report_to="wandb" --log_name=$env:LOG_NAME `
  --gradient_accumulation_steps=1 `
  --resume_from_checkpoint="latest" `
  --guidance_scale=7.5 2>&1 | Tee-Object -FilePath $env:OUTPUT_LOG

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
