#!/usr/bin/env bash
#SBATCH --job-name=sample-rnn-smiles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=H100:1
#SBATCH --gpus-per-task=H100:1
#SBATCH --mem=50G
#SBATCH --partition=long
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/ps/home/CORP/yassir.elmesbahi/project/SMILES-RNN/out/sample-rnn-smiles.out
#SBATCH --error=/mnt/ps/home/CORP/yassir.elmesbahi/project/SMILES-RNN/out/sample-rnn-smiles.out     



### GENERAL
export CUDA_HOME='/cm/shared/apps/cuda12.1/toolkit/12.1.1'
export FORCE_CUDA="1"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

### CONFIGURATIONS
### JOBS
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NODE_RANK=$SLURM_NODEID
export N_NODES=$SLURM_NNODES
export GPUS_PER_NODE=${SLURM_GPUS_PER_NODE#*:}
export NUM_PROCS=$((N_NODES * GPUS_PER_NODE))
export WORLD_SIZE=$NUM_PROCS
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%60001+5110)) # Port must be 0-65535

### DEBUG
#export NCCL_DEBUG=INFO
#export TORCH_CPP_LOG_LEVEL=INFO 
#export TORCH_DISTRIBUTED_DEBUG=INFO
#export NCCL_P2P_DISABLE=1
#export TORCH_SHOW_CPP_STACKTRACES=1
#export TORCH_LOGS="+dynamo"
#export TORCHDYNAMO_VERBOSE=1

### NCCL DEBUG
#export NCCL_LL_THRESHOLD=0
#export NCCL_SOCKET_IFNAME=ens3
#export NCCL_SOCKET_IFNAME=eth4
#export NCCL_SOCKET_IFNAME=lo

### PATHS
export HOME_DIR="/mnt/ps/home/CORP/yassir.elmesbahi"
export PROJ_NAME="SMILES-RNN"
export PROJ_DIR="${HOME_DIR}/project/${PROJ_NAME}"
export SANDBOX_DIR="${HOME_DIR}/sandbox"
export DATA_DIR="${HOME_DIR}/ondemand/data/moses"
export HF_HOME="${SANDBOX_DIR}/.hf_home"
export RUNNER="${PROJ_DIR}/scripts/sample_model.py"

# In a SLURM job, you CANNOT use `conda activate` and instead MUST use:
#source ${HOME_DIR}/.bashrc
#source ${HOME_DIR}/miniforge3/etc/profile.d/conda.sh
#mamba activate dev


export N_SAMPLES=10000
export TEMPERATURE=1.
export GTR_ARGS=" \
    --n_heads ${N_HEAD} \
    --n_dims ${N_DIMS} \
    --ff_dims ${FF_DIMS} \
    --n_layers ${N_LAYERS} \
    --dropout ${DROPOUT} \
    --learning_rate ${LR} \
    "


### EVALUATION PARAMETERS




# --unique ??

export PYTHON_LAUNCHER="python \
"

export TORCH_LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $N_NODES \
    --rdzv_backend static \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --max_restarts 0 \
    --node_rank $NODE_RANK \
"


export EPOCH=5
export MODEL="RNN"
export DEVICE="gpu"
export WAIT_TIME=5


for CELL_TYPE in 'gru' 'lstm'; do
    for SUBGRAMMAR in  'safe-hr' 'safe-rotatable' 'safe-brics' 'safe-recap' 'safe-mmpa'; do
        export ARCHITECTURE="${MODEL}_${CELL_TYPE}"
        export MODEL_PATH="${SANDBOX_DIR}/models/${ARCHITECTURE}_${SUBGRAMMAR}/Prior_None_Epoch-${EPOCH}.ckpt"
        export OUTPUT_DIR="${SANDBOX_DIR}/models/${ARCHITECTURE}_${SUBGRAMMAR}/sampling"

        echo "###### Sampling with $MODEL (${CELL_TYPE}) architecture..."
        echo ">>> Using SAFE with '$SUBGRAMMAR' slicer "

        for SEED in 19 33 56 76 99; do
            echo "===>>> Sampling with seed=$SEED ..."
            
            export OUTPUT_FILE="${OUTPUT_DIR}/predictions_${N_SAMPLES}_${SEED}.txt"
            export RUNNER_ARGS=" \
                --path ${MODEL_PATH} \
                --model ${MODEL} \
                --output ${OUTPUT_FILE} \
                --device ${DEVICE} \
                --number ${N_SAMPLES} \
                --temperature ${TEMPERATURE} \
                --seed ${SEED} \
                --native \
            "
            export CMD="${PYTHON_LAUNCHER} ${RUNNER} ${RUNNER_ARGS}"
            echo "===>>> Running command '${CMD}'"
            $CMD
            echo ">> Waiting ${WAIT_TIME} seconds..."
            sleep $WAIT_TIME
        done
    done
done
