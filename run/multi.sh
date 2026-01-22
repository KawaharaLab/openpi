#!/bin/bash
#PBS -q debug-g
#PBS -l select=4:ncpus=72:mpiprocs=1
#PBS -W group_list=gr41
#PBS -j oe

module purge
module load nvidia nv-hpcx
module load hdf5
cd "$PBS_O_WORKDIR"

# Threading/BLAS settings for each rank
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}

MASTER_ADDR=$(getent hosts "$(head -n1 "$PBS_NODEFILE")" | awk '{print $1; exit}')
MASTER_PORT=29500
NNODES=$(sort -u "$PBS_NODEFILE" | wc -l)
RUN_NAME=${RUN_NAME:-distributed_test_$(date +%s)}

# IF自動検出（見つからなければ未設定でGloo/NCCLに任せる）
IFACE=$(ip -o route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") {print $(i+1); exit}}')
if [ -n "$IFACE" ]; then
  export NCCL_SOCKET_IFNAME=$IFACE
  export GLOO_SOCKET_IFNAME=$IFACE
fi
export MASTER_ADDR MASTER_PORT NNODES
unset OMPI_MCA_mca_base_env_list

# wandb: キーがあればオンライン、無ければオフライン
if [ -n "${WANDB_API_KEY:-}" ]; then
  unset WANDB_MODE
else
  export WANDB_MODE=offline
fi

# ランクごとにGPU/torch確認（デバッグ用）
mpiexec -np ${NNODES} --map-by ppr:1:node:PE=${OMP_NUM_THREADS} --bind-to core --report-bindings --hostfile "$PBS_NODEFILE" \
  bash -lc "module purge; module load nvidia nv-hpcx; module load hdf5; \
            nvidia-smi --query-gpu=name,uuid --format=csv,noheader; \
            source .venv/bin/activate; \
            python -c 'import os, socket, torch; print(socket.gethostname(), torch.cuda.is_available(), torch.cuda.device_count(), os.cpu_count(), len(os.sched_getaffinity(0)))'"

# 本番実行
mpiexec -np ${NNODES} --map-by ppr:1:node:PE=${OMP_NUM_THREADS} --bind-to core --report-bindings --hostfile "$PBS_NODEFILE" \
  -x OMP_NUM_THREADS -x MKL_NUM_THREADS \
  -x MASTER_ADDR -x MASTER_PORT -x NCCL_SOCKET_IFNAME -x GLOO_SOCKET_IFNAME \
  -x CUDA_VISIBLE_DEVICES -x PATH -x LD_LIBRARY_PATH -x WANDB_MODE -x WANDB_API_KEY \
  bash -lc "
    module purge; module load nvidia nv-hpcx; module load hdf5
    cd $PBS_O_WORKDIR
    source .venv/bin/activate
    NODE_RANK=\$OMPI_COMM_WORLD_RANK
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=0
    torchrun \
      --nnodes=${NNODES} \
      --nproc_per_node=1 \
      --node_rank=\${NODE_RANK} \
      --master_addr=${MASTER_ADDR} \
      --master_port=${MASTER_PORT} \
      scripts/train_pytorch.py pi0_ur3_robotiq_ft \
        --exp_name ${RUN_NAME} \
        --batch_size 128 \
            --num_workers 16 \
        --no-pytorch-gradient-checkpointing \
            --freeze_pretrained_steps 10000 \
            --num_train_steps 40000 \
        --save_interval 10000
  "