CUDA_VISIBLE_DEVICES=0 python runners/evaluation_tracking.py \
--score_model_dir ScoreNet/ckpt_genpose.pth \
--energy_model_dir EnergyNet/ckpt_genpose.pth \
--data_path NOCS_DATASET_PATH \
--sampler_mode ode \
--max_eval_num 1000000 \
--percentage_data_for_test 1.0 \
--batch_size 256 \
--seed 0 \
--test_source aligned_real_test \
--result_dir results \
--eval_repeat_num 50 \
--pooling_mode average \
--ranker energy_ranker \
--T0 0.15 \
# --save_video \
