#!/usr/bin/env bash
#SBATCH --job-name=train-rnn-gru-safe-recap-100000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=H100:1
#SBATCH --gpus-per-task=H100:1
#SBATCH --mem=50G
#SBATCH --partition=long
#SBATCH --time=24:00:000
#SBATCH --output=/mnt/ps/home/CORP/yassir.elmesbahi/project/smiles-rnn/out/train-rnn-gru-safe-recap-100000.out
#SBATCH --error=/mnt/ps/home/CORP/yassir.elmesbahi/project/smiles-rnn/out/train-rnn-gru-safe-recap-100000.out

export N_SAMPLES=100000

export MODEL="RNN"
#export CELL_TYPE="lstm"
export CELL_TYPE="gru"
export ARCHITECTURE="${MODEL}_${CELL_TYPE}"
export DATASET_TYPE="moses"

#export GRAMMAR="SMILES"
#export SUBGRAMMAR="smiles"
export GRAMMAR="SAFE"
#export SUBGRAMMAR="safe-hr"
#export SUBGRAMMAR="safe-rotatable"
#export SUBGRAMMAR="safe-brics"
#export SUBGRAMMAR="safe-mmpa"
export SUBGRAMMAR="safe-recap"
export SLICER="${SUBGRAMMAR#*-}"

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
export PROJ_NAME="smiles-rnn"
export PROJ_DIR="${HOME_DIR}/project/${PROJ_NAME}"
export SANDBOX_DIR="${HOME_DIR}/sandbox"
export DATA_DIR="${HOME_DIR}/ondemand/data"
export HF_HOME="${SANDBOX_DIR}/.hf_home"
export RUNNER="${PROJ_DIR}/scripts/train_prior.py"

# In a SLURM job, you CANNOT use `conda activate` and instead MUST use:
#source ${HOME_DIR}/.bashrc
#source ${HOME_DIR}/miniforge3/etc/profile.d/conda.sh
#mamba activate dev

declare -A LR_MAP=(
    [10000]=1e-3
    [100000]=1e-3
)

declare -A EPOCH_MAP=(
    [10000]=3
    [100000]=6
)

### RNN PARAMETERS
export LAYER_SIZE=512
export NUM_LAYERS=3
export EMB_LAYER_SIZE=256
export DROPOUT=0.0
export RNN_ARGS=" \
    --layer_size ${LAYER_SIZE} \
    --num_layers ${NUM_LAYERS} \
    --cell_type ${CELL_TYPE} \
    --embedding_layer_size ${EMB_LAYER_SIZE} \
    --dropout ${DROPOUT} \
    --learning_rate ${LR_MAP[$N_SAMPLES]} \
"

### TRAINING PARAMETERS

export TRAIN_SMILES="${DATA_DIR}/${DATASET_TYPE}/${SUBGRAMMAR}_${N_SAMPLES}-train.smi"
export VALID_SMILES="${DATA_DIR}/${DATASET_TYPE}/${SUBGRAMMAR}_${N_SAMPLES}-valid.smi"
export TEST_SMILES="${DATA_DIR}/${DATASET_TYPE}/${SUBGRAMMAR}_${N_SAMPLES}-test.smi"


export OUTPUT_DIR="${SANDBOX_DIR}/models/${ARCHITECTURE}_${SUBGRAMMAR}_${N_SAMPLES}"
export SUFFIX="Moses"
export VALIDATE_FREQUENCY=500
export BATCH_SIZE=128
export DEVICE="gpu"


declare -A ARGS_CFG_MAP=(
    ["RNN"]="${RNN_ARGS}"
    ["Transformer"]="${TRANSFORMER_ARGS}"
    ["GTr"]="${GTR_ARGS}"
)

#export CHKPT_DIR="${OUTPUT_DIR}/checkpoint-30000"
export RUNNER_ARGS=" \
    --train_smiles ${TRAIN_SMILES} \
    --output_directory ${OUTPUT_DIR} \
    --suffix ${SUFFIX} \
    --grammar ${GRAMMAR} \
    --slicer ${SLICER} \
    --valid_smiles ${VALID_SMILES} \
    --test_smiles ${TEST_SMILES} \
    --validate_frequency ${VALIDATE_FREQUENCY} \
    --n_epochs ${EPOCH_MAP[$N_SAMPLES]} \
    --batch_size ${BATCH_SIZE} \
    --device ${DEVICE} \
    ${MODEL} \
"

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


# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="${PYTHON_LAUNCHER} ${RUNNER} ${RUNNER_ARGS} ${ARGS_CFG_MAP[$MODEL]}" 
echo "===>>> Running command '${CMD}'"
#srun --jobid $SLURM_JOBID --export=ALL  $CMD
$CMD

