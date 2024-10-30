export NCCL_SOCKET_IFNAME=hsn
export CXI_FORK_SAFE="1"
export FI_CXI_DISABLE_CQ_HUGETLB="1"
export FI_MR_CACHE_MONITOR="userfaultfd"
export CXI_FORK_SAFE_HP="1"
export NCCL_CROSS_NIC="1"
export FI_CXI_DISABLE_HOST_REGISTER="1"
export FI_HMEM_CUDA_USE_GDRCOPY=1
export FI_HMEM_CUDA_ENABLE_XFER=1

export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL="PHY"
export NCCL_CUMEM_ENABLE=0

nsys profile -t cuda,nvtx,osrt,cublas,cudnn --stats=true --output=$SCRATCH/gpt \
     --force-overwrite true --cuda-memory-usage=true --cudabacktrace=all \
     --gpu-metrics-device=all --nic-metrics=true --capture-range=cudaProfilerApi \
     --capture-range-end=stop torchrun --nproc-per-node 4  scripts/train_llama.py -d $SCRATCH/data/c4-realnewslike/ -dp 1 -pp 4

# if [ ${PROFILE} -eq 1 ]; then
#     PROFILE_CMD="nsys profile -w true --trace=cuda,nvtx,osrt,cudnn,cublas,mpi \
#                 --cudabacktrace=true -x true --gpu-metrics-device=all \
#                 --sample=none --force-overwrite=true --nic-metrics=true \
#                 --duration=120 --capture-range=cudaProfilerApi \
#                 --output=./log/profile.nsys-rep"
# fi

# ${PROFILE_CMD} torchrun --nnodes=1 --nproc_per_node=4 $HOME/topal-internship/example.py

