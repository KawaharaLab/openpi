#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:30:00
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
# python examples/ur3_robotiq/convert_lan_to_lerobot_cartesian.py
# python examples/ur3_robotiq/convert_lan_to_lerobot.py
# python ./scripts/compute_norm_stats.py --config-name pi0_ur3_robotiq_ft
# python ./scripts/compute_norm_stats.py --config-name pi0_ur3_robotiq_ft_angles
python ./scripts/train_pytorch.py pi0_ur3_robotiq_ft --exp-name trial --batch-size 128 --freeze-pretrained-steps 200 --num_train_steps 200 --save_interval 10000
exit 0
