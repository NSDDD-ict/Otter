source /workspace/S/zhangyang/miniconda3/bin/activate PromptCBLUE
conda info -e
cd /lustre/S/zhangyang/chengshuang/LLM/Otter
export  NCCL_IB_DISABLE=1 

accelerate launch --config_file=/lustre/S/zhangyang/chengshuang/LLM/Otter/pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
    /lustre/S/zhangyang/chengshuang/LLM/Otter/pipeline/train/instruction_following.py \
    --pretrained_model_name_or_path=/lustre/S/zhangyang/chengshuang/LLM/Otter/OTTER-Video-LLaMA7B-DenseCaption \
    --mimicit_path="/lustre/S/zhangyang/chengshuang/LLM/Otter/output/FunQAf128_instructions_train.json" \
    --images_path="/lustre/S/zhangyang/chengshuang/LLM/Otter/output/FunQA.json" \
    --train_config_path="/lustre/S/zhangyang/chengshuang/LLM/Otter/output/FunQAf128_train_train.json" \
    --batch_size=8 \
    --num_epochs=5 \
    --report_to_wandb \
    --wandb_entity=ljunius \
    --run_name=otter9B_funqa_icl \
    --wandb_project=MLLM \
    --workers=32 \
    --lr_scheduler=cosine \
    --learning_rate=1e-5 \
    --warmup_steps_ratio=0.01 \
    --save_ckpt_each_epoch \
    --gradient_checkpointing \
    --gradient_accumulation_steps=4 \
    --save_ckpt_each_epoch \
    --external_save_dir=/lustre/S/zhangyang/chengshuang/LLM/Otter/exp_result \
    --offline
