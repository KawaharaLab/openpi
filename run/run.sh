#!/bin/bash
#PBS -q short-g
#PBS -l select=1
#PBS -W group_list=gr41


# load modules
module purge
module load nvidia
module load hdf5

# job execution
cd $PBS_O_WORKDIR
export WANDB_API_KEY="c85b817c62f441243d232b381088358e72fa2b19"
# nvidia-smi
source .venv/bin/activate
# python scripts/train_pytorch.py pi0_ur3_robotiq --exp-name full_trial --reinit-action-expert
# python scripts/train_pytorch.py pi0_ur3_robotiq --exp-name full_fix
# python ./scripts/train_pytorch.py pi0_ur3_robotiq_cartesian_pos --exp-name full_trial
# python ./scripts/train_pytorch.py pi0_ur3_robotiq_cartesian_pos_ft_lora --exp-name trial --reinit-action-expert
python ./scripts/train_pytorch.py pi0_ur3_robotiq_ft_lora --exp-name full --batch-size 32 --freeze-pretrained-steps 4000
# python examples/ur3_robotiq/convert_lan_to_lerobot_cartesian.py
# python examples/ur3_robotiq/convert_lan_to_lerobot.py
# python ./scripts/compute_norm_stats.py --config-name pi0_ur3_robotiq_cartesian_pos_ft_lora
# python ./scripts/compute_norm_stats.py --config-name pi0_ur3_robotiq_ft_angles
exit 0
