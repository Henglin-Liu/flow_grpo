export CUDA_VISIBLE_DEVICES=0
# 1 GPU
# python scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_1gpu
# 4 GPU
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_4gpu
# CUDA_VISIBLE_DEVICES=0,1 python -m debugpy --wait-for-client --listen 5680 scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_1gpu

# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29549 \
#     -m debugpy --wait-for-client --listen 5680 scripts/train_consisID.py --config config/grpo.py:consisID_1gpu
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29401 -m pdb scripts/train_consisID.py --config config/grpo.py:consisID_1gpu


# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29549 \
#     -m debugpy --wait-for-client --listen 5680 scripts/train_consisID.py --config config/grpo.py:consisID_1gpu

# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=2 --main_process_port 29401 scripts/train_consisID.py --config config/grpo.py:consisID_2gpu