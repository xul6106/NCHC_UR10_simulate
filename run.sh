# =========================================== Dataset download =================================================
# /home/user/miniconda3/envs/robomimic/bin/python /scripts/download_datasets.py \
# --tasks lift --dataset_types ph --hdf5_types image



# ========================================== collect with rewards ==============================================
# 收數據(UR10, dualsense, bottles)：1754894280_038122(0811下午_grab)、1754991091_713582(0812下午_PointTo_keyboard)
/home/user/miniconda3/envs/robomimic/bin/python scripts/collect_human_demonstrations_rewards.py \
--directory /home/user/ur10_demonstrations --environment DualTableTask \
--robots UR10 --device dualsense --renderer mjviewer --camera agentview


# # Convert_robosuite(Split_train_val)：
# /home/user/miniconda3/envs/robomimic/bin/python /scripts/conversion/convert_robosuite.py \
# --dataset /home/user/ur10_demonstrations/combined_dataset_800.hdf5


# # Dataset_States_to_obs：
# /home/user/miniconda3/envs/robomimic/bin/python /scripts/dataset_states_to_obs.py \
# --dataset /home/user/ur10_demonstrations/combined_dataset_800.hdf5 \
# --output_name /home/user/ur10_demonstrations/combined_800_image.hdf5 \
# --done_mode 2 \
# --copy_rewards \
# --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84


# # Playback(HDF5)：
# /home/user/miniconda3/envs/robomimic/bin/python /scripts/playback_dataset.py \
# --dataset /home/user/ur10_demonstrations/1755496167_5995195/low_dim.hdf5 \
# --render_image_names frontview \
# --video_path /home/user/ur10_demonstrations/1755496167_5995195/playback_dataset.mp4



# ================================================ Training =====================================================
# # Train(lift)：
# /home/user/miniconda3/envs/robomimic/bin/python /scripts/train.py \
# --config /bc_rnn_0819.json \
# --dataset /home/user/ur10_demonstrations/combined_800_image.hdf5


# # Playback：
# /home/user/miniconda3/envs/robomimic/bin/python /scripts/run_trained_agent.py \
# --agent /home/user/robomimic/bc_trained_models/core/bc_rnn/lift/ph/low_dim/trained_models/core_bc_rnn_lift_ph_low_dim/20250821223611/models/model_epoch_500.pth \
# --n_rollouts 50 --horizon 400 --seed 0 \
# --video_path /home/user/robomimic/bc_trained_models/core/bc_rnn/lift/ph/low_dim/trained_models/core_bc_rnn_lift_ph_low_dim/20250821223611/videos/playback_500.mp4 \
# --camera_names frontview


# # Tensorboard：
# tensorboard --logdir /home/user/robomimic/bc_trained_models/core/bc_rnn/lift/ph/low_dim/trained_models/core_bc_rnn_lift_ph_low_dim/20250821223611/logs/tb --bind_all








# ======================= classic way to collect data =================

# # 收數據(UR10, dualsense, bottles, classic way)：
# /home/user/miniconda3/envs/robomimic/bin/python scripts/collect_human_demonstrations.py \
# --directory /home/user/ur10_demonstrations --environment DualTableTask \
# --robots UR10 --device dualsense --renderer mjviewer --camera agentview


# # Robosuite_add_absolute_actions
# /home/user/miniconda3/envs/robomimic/bin/python /scripts/conversion/robosuite_add_absolute_actions.py \
# --dataset /home/user/ur10_demonstrations/1755166817_116234/demo.hdf5


# # Convert_robosuite
# /home/user/miniconda3/envs/robomimic/bin/python /scripts/conversion/convert_robosuite.py \
# --dataset /home/user/ur10_demonstrations/1755166817_116234/demo.hdf5


