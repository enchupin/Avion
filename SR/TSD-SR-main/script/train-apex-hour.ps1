$ErrorActionPreference = "Stop"

$env:MODEL_NAME = "checkpoint/tsdsr"
$env:TEACHER_MODEL_NAME = "checkpoint/teacher/"
$env:DEFAULT_EMBED = "dataset/default"
$env:NULL_EMBED = "dataset/null"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:WANDB_MODE = "offline"
$env:OUTPUT_DIR = "checkpoint/tsdsr-save-apex-hour/"
$env:OUTPUT_LOG = "logs/tsdsr_apex_hour.log"
$env:LOG_NAME = "tsdsr-apex-hour"

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$ErrorActionPreference = "Continue"
accelerate launch --num_machines 1 --dynamo_backend "no" --num_processes 1 --mixed_precision="fp16" train/train.py `
  --pretrained_model_name_or_path=$env:MODEL_NAME `
  --teacher_lora_path=$env:TEACHER_MODEL_NAME `
  --train_data_dir data/APEX_train_384 `
  --default_embedding_dir=$env:DEFAULT_EMBED --null_embedding_dir=$env:NULL_EMBED `
  --train_batch_size=1 --rank=64 --rank_vae=16 --rank_lora=16 `
  --num_train_epochs=1 --checkpointing_steps=120 --validation_steps=1000 --max_train_steps=720 `
  --learning_rate=5e-06 --learning_rate_reg=1e-06 --lr_scheduler="cosine_with_restarts" --lr_warmup_steps=72 `
  --seed=43 --use_default_prompt --use_teacher_lora --use_random_bias --gradient_checkpointing `
  --output_dir=$env:OUTPUT_DIR `
  --report_to="wandb" --log_name=$env:LOG_NAME `
  --gradient_accumulation_steps=1 `
  --resume_from_checkpoint="latest" `
  --guidance_scale=7.5 2>&1 | Tee-Object -FilePath $env:OUTPUT_LOG

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
