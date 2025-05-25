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
bash ./xtuner-train-rm-job.sh"

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
--job-hang-interval=1200 --enable-c4d-hang-detection=True --enable-perftracker=True" \
    --node_names "e01-cn-0gx41zkg82i,\
e01-cn-bcd41krww24,\
e01-cn-0gx41zkg82s,\
e01-cn-bcd41krww2o,\
e01-cn-0gx41zkg86o,\
e01-cn-bcd41krww2y,\
e01-cn-bcd41krww0g,\
e01-cn-0gx41zkg814,\
e01-cn-0gx41zkg850,\
e01-cn-k9642dfwm03,\
e01-cn-bcd41krww2n,\
e01-cn-bcd41krww3i,\
e01-cn-0gx41zkg80a,\
e01-cn-0gx41zkg82h,\
e01-cn-0gx41zkg85j,\
e01-cn-0gx41zkg80j,\
e01-cn-0gx41zkg859,\
e01-cn-0gx41zkg86n,\
e01-cn-bcd41krww1u,\
e01-cn-0gx41zkg83b,\
e01-cn-g4t48g3se0a,\
e01-cn-0gx41zkg84f,\
e01-cn-bcd41krww0q,\
e01-cn-bcd41krww4b,\
e01-cn-lf640yjej0l,\
e01-cn-lf640yjej0k,\
e01-cn-zp548g5qv02,\
e01-cn-0gx41zkg812,\
e01-cn-0gx41zkg83a,\
e01-cn-0gx41zkg862,\
e01-cn-0gx41zkg81c,\
e01-cn-0gx41zkg83k,\
e01-cn-0gx41zkg85r,\
e01-cn-0gx41zkg86l,\
e01-cn-bcd41krww2m,\
e01-cn-0gx41zkg807,\
e01-cn-0gx41zkg81l,\
e01-cn-bcd41krww3r,\
e01-cn-0gx41zkg839,\
e01-cn-0gx41zkg84x,\
e01-cn-0gx41zkg825,\
e01-cn-0gx41zkg82z,\
e01-cn-bcd41krww0z,\
e01-cn-0gx41zkg86b,\
e01-cn-0gx41zkg81b,\
e01-cn-bcd41krww0e,\
e01-cn-0gx41zkg81v,\
e01-cn-bcd41krww2c,\
e01-cn-bcd41krww18,\
e01-cn-bcd41krww40,\
e01-cn-bcd41krww1i,\
e01-cn-0gx41zkg82e,\
e01-cn-0gx41zkg83s,\
e01-cn-bcd41krww21,\
e01-cn-bcd41krww1s,\
e01-cn-bcd41krww3q,\
e01-cn-0gx41zkg81k,\
e01-cn-bcd41krww0y,\
e01-cn-bcd41krww2b,\
e01-cn-0w741zkvu05,\
e01-cn-0w741zkvu0p,\
e01-cn-kvw3z9b7f02,\
e01-cn-0w741zkvu0y,\
e01-cn-0w741zkvu0e,\
e01-cn-0w741zkvu1r,\
e01-cn-0w741zkvu1z,\
e01-cn-0w741zkvu01,\
e01-cn-0w741zkvu0v,\
e01-cn-0w741zkvu15,\
e01-cn-0w741zkvu0b,\
e01-cn-0w741zkvu1e,\
e01-cn-0w741zkvu26,\
e01-cn-zim40yjn30z,\
e01-cn-0w741zkvu08"
}

commit
