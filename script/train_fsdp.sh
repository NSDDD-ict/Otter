accelerate launch --config_file=pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
    pipeline/train/instruction_following.py \
    --pretrained_model_name_or_path=/data/chengshuang/Otter_origin/model_weight/models--luodian--OTTER-9B-INIT/snapshots/cc075926603ab1ffdef5f0a7809f84201ec31346 \
    --mimicit_path="/data/chengshuang/Otter/output/FunQAf128_instructions_train.json" \
    --images_path="/data/chengshuang/Otter/output/FunQA.json" \
    --train_config_path="/data/chengshuang/Otter/output/FunQAf128_train_train.json" \
    --batch_size=8 \
    --num_epochs=5 \
    --report_to_wandb \
    --wandb_entity=ljunius \
    --run_name=otter9B_funqa_icl \
    --wandb_project=MLLM \
    --workers=8 \
    --lr_schueduler=cosine \
    --learning_rate=1e-5 \
    --warmup_steps_ratio=0.01 \
    --save_ckpt_each_epoch \
    --gradient_checkpointing \
    --gradient_accumulation_steps=4 \
    --save_ckpt_each_epoch \
    --external_save_dir=/data/chengshuang/Otter/exp_result \
    # --offline