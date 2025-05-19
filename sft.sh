#!/bin/bash
# hs: ws1lu4iyv5yjjyvp
# ddd: ws1ujefpjyfgqjwp

export HOME="/cpfs01/shared/alillm_hs/zouyicheng/"


function commit {
    num_nodes=2
    name="xtuner-train-rm-internlm2_5_7b"
    num_tasks_per_node=1
    node_cpus=96
    num_gpus=8
    node_mems=1280Gi

    cmd=". /cpfs01/shared/alillm_hs/zouyicheng/.bashrc && conda activate xtuner && \
    cd /cpfs01/shared/alillm_hs/zouyicheng/xtuner && \
    export PYTHONPATH=/cpfs01/shared/alillm_hs/zouyicheng/xtuner && \
    bash ./xtuner-sft-rm-job.sh"

    /cpfs01/shared/alillm_hs/zouyicheng/dlc create job --config /cpfs01/shared/alillm_hs/zouyicheng/dlc.config \
    --name $name \
    --worker_count $num_nodes \
    --kind PyTorchJob \
    --worker_cpu $node_cpus \
    --worker_gpu $num_gpus \
    --worker_memory $node_mems \
    --workspace_id ws1lu4iyv5yjjyvp \
    --data_sources data1ewbw1ztmmyh,data1bgvj0n14to0,datasnlgjr5gyk0c,data1ubhj4714msc,data9qbxtzsqaa1f,data1o8qdjce0kd0 \
    --worker_image pjlab-shanghai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/pjlab-eflops/lishuaibin:lishuaibin-xpuyu-trainrlhf \
    --worker_shared_memory 128Gi \
    --command "$cmd" \
    --priority 9 \
    --aimaster_enable true \
    --aimaster_args="--job-execution-mode=Sync \
--enable-job-restart=True --max-num-of-job-restart=10 \
--job-restart-timeout=1800 --fault-tolerant-policy=OnFailure \
--enable-local-detection=True --enable-job-hang-detection=True \
--job-hang-interval=1200"
}

commit
