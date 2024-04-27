GPU_NUMBER=1
MODEL_NAME='joelniklaus/legal-croatian-roberta-base'
LOWER_CASE='True'
BATCH_SIZE=2
ACCUMULATION_STEPS=16
TASK='croatian-lex'
#第一次的log在max256里
Note='multilex-max256'
lr=3e-5
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python myexperiments/multilex.py --max_segments 64 32 16   --max_seg_length 64 128 256 --curriculum False --convolutional True --model_name_or_path ${MODEL_NAME} --do_lower_case ${LOWER_CASE} --task ${TASK} --output_dir logs/${Note}/${TASK}/${lr}/${MODEL_NAME}/seed_1  --do_train --do_eval --do_pred --overwrite_output_dir --load_best_model_at_end --metric_for_best_model micro-f1 --greater_is_better True --evaluation_strategy epoch --save_strategy epoch --save_total_limit 3 --num_train_epochs 20 --learning_rate ${lr} --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --seed 1 --fp16 --fp16_full_eval --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
Note='multilex-max512'
lr=1e-5
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python myexperiments/multilex.py --max_segments 32 16 8  --max_seg_length 128 256 512 --curriculum False --convolutional True --model_name_or_path ${MODEL_NAME} --do_lower_case ${LOWER_CASE} --task ${TASK} --output_dir logs/${Note}/${TASK}/${lr}/${MODEL_NAME}/seed_1  --do_train --do_eval --do_pred --overwrite_output_dir --load_best_model_at_end --metric_for_best_model micro-f1 --greater_is_better True --evaluation_strategy epoch --save_strategy epoch --save_total_limit 3 --num_train_epochs 20 --learning_rate ${lr} --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --seed 1 --fp16 --fp16_full_eval --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
