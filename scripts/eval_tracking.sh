CUDA_VISIBLE_DEVICES=3 python runners/evaluation_tracking.py \
--score_model_dir ScoreNet/ckpt_genpose.pth \
--energy_model_dir EnergyNet/ckpt_genpose.pth \
--data_path /root/autodl-tmp/jiyaozhang/AnyPose/data/NOCS_DATA/nocs_cvpr_19 \
--sampler_mode ode \
--max_eval_num 1000000 \
--percentage_data_for_test 1.0 \
--batch_size 256 \
--seed 0 \
--result_dir results \
--test_source aligned_real_test \
--eval_repeat_num 50 \
--pooling_mode average \
--ranker energy_ranker \
--T0 0.15 \
# --save_video \
# --data_path NOCS_DATASET_PATH \
