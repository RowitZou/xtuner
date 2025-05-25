#!/bin/bash
# hs: ws1lu4iyv5yjjyvp
# ddd: ws1ujefpjyfgqjwp

export HOME="/cpfs01/shared/alillm_hs/zouyicheng/"


function commit {
    num_nodes=57
    name="xtuner-train-rm-internlm2_5_7b"
    num_tasks_per_node=1
    node_cpus=96
    num_gpus=8
    node_mems=1280Gi

    cmd=". /cpfs01/shared/alillm_hs/zouyicheng/.bashrc && conda activate /rm_train && \
cd /cpfs01/shared/alillm_hs/zouyicheng/xtuner && \
export PYTHONPATH=/cpfs01/shared/alillm_hs/zouyicheng/xtuner && \
export NCCL_IB_TC=136 && \
export NCCL_IB_SL=5 && \
export NCCL_IB_GID_INDEX=3 && \
export NCCL_SOCKET_IFNAME=bond1 && \
export NCCL_DEBUG=INFO && \
export NCCL_IB_HCA=mlx5_bond && \
export NCCL_IB_TIMEOUT=22 && \
export NCCL_IB_QPS_PER_CONNECTION=8 && \
export NCCL_MIN_NCHANNELS=4 && \
export NCCL_NET_PLUGIN=none && \
export ACCL_C4_STATS_MODE=CONN && \
export ACCL_IB_SPLIT_DATA_NUM=4 && \
export ACCL_IB_QPS_LOAD_BALANCE=1 && \
export ACCL_IB_GID_INDEX_FIX=1 && \
export ACCL_LOG_TIME=1 && \
bash ./xtuner-debug-rm-job.sh"

    /cpfs01/shared/alillm_hs/zouyicheng/dlc create job --config /cpfs01/shared/alillm_hs/zouyicheng/dlc.config \
    --name $name \
    --worker_count $num_nodes \
    --kind PyTorchJob \
    --worker_cpu $node_cpus \
    --worker_gpu $num_gpus \
    --worker_memory $node_mems \
    --workspace_id ws1ujefpjyfgqjwp \
    --data_sources data1ewbw1ztmmyh,data1bgvj0n14to0,datasnlgjr5gyk0c,data1ubhj4714msc,data9qbxtzsqaa1f,data1o8qdjce0kd0 \
    --worker_image pjlab-shanghai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/pjlab-eflops/zouyicheng:zouyicheng-xtuner-accl-perftracker \
    --worker_shared_memory 128Gi \
    --command "$cmd" \
    --priority 9 \
    --aimaster_enable true \
    --aimaster_args="--job-execution-mode=Sync \
--enable-job-restart=True --max-num-of-job-restart=30 \
--job-restart-timeout=1800 --fault-tolerant-policy=OnFailure \
--enable-local-detection=True --enable-job-hang-detection=True \
--job-hang-interval=1200 --enable-c4d-hang-detection=True --enable-perftracker=True"
}

commit
