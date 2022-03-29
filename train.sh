set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0,1,3,4

python3 main.py --env BreakoutNoFrameskip-v4 --case atari --opr train --force \
  --num_gpus 4 --num_cpus 32 --cpu_actor 8 --gpu_actor 8 --p_mcts_num 1 \
  --seed 0 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info 'EfficientZero-V1' \
  --save_video
