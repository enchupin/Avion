export MODEL_NAME="checkpoint/tsdsr";
export TEACHER_MODEL_NAME="checkpoint/teacher/";
export CHECKPOINT_PATH="checkpoint/tsdsr";
export DEFAULT_EMBED="dataset/default"
export NULL_EMBED="dataset/null";
export HF_ENDPOINT="https://hf-mirror.com";
export WANDB_MODE="offline";
export OUTPUT_DIR="checkpoint/tsdsr-save/";
export OUTPUT_LOG="logs/tsdsr.log";
export LOG_NAME="tsdsr-train";
accelerate launch --num_processes 1 --mixed_precision="fp16" train/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --teacher_lora_path=$TEACHER_MODEL_NAME \
  --train_data_dir data/DIV2K/train \
  --default_embedding_dir=$DEFAULT_EMBED --null_embedding_dir=$NULL_EMBED \
  --train_batch_size=1 --rank=64 --rank_vae=64 --rank_lora=64  \
  --num_train_epochs=1 --checkpointing_steps=100 --validation_steps=100  --max_train_steps=300 \
  --learning_rate=5e-06  --learning_rate_reg=1e-06 --lr_scheduler="cosine_with_restarts" --lr_warmup_steps=30 \
  --seed=43 --use_default_prompt --use_teacher_lora --use_random_bias --gradient_checkpointing \
  --output_dir=$OUTPUT_DIR \
  --report_to="wandb" --log_code --log_name=$LOG_NAME \
  --gradient_accumulation_steps=1 \
  --resume_from_checkpoint="latest" \
  --guidance_scale=7.5  2>&1 | tee $OUTPUT_LOG
