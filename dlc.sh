#!/bin/bash
# hs: ws1lu4iyv5yjjyvp
# ddd: ws1ujefpjyfgqjwp

export HOME="/cpfs01/shared/llm_ddd/zouyicheng/"


function commit {
    num_nodes=2
    name="xtuner-train-rm"
    num_tasks_per_node=1
    node_cpus=96
    num_gpus=8
    node_mems=1024Gi

    cmd=". /cpfs01/shared/llm_ddd/zouyicheng/.bashrc && conda activate xtuner && \
    cd /cpfs01/shared/llm_ddd/zouyicheng/xtuner && \
    export PYTHONPATH=/cpfs01/shared/llm_ddd/zouyicheng/xtuner && \
    bash ./xtuner-train-rm-job.sh"

    /cpfs01/shared/llm_ddd/zouyicheng/dlc create job --config /cpfs01/shared/llm_ddd/zouyicheng/dlc.config \
    --name $name \
    --worker_count $num_nodes \
    --kind PyTorchJob \
    --worker_cpu $node_cpus \
    --worker_gpu $num_gpus \
    --worker_memory $node_mems \
    --workspace_id ws1ujefpjyfgqjwp \
    --data_sources data1ewbw1ztmmyh,data1bgvj0n14to0,datasnlgjr5gyk0c,data1ubhj4714msc,data9qbxtzsqaa1f \
    --worker_image pjlab-shanghai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/pjlab-eflops/lishuaibin:lishuaibin-xpuyu-trainrlhf \
    --worker_shared_memory 128Gi \
    --command "$cmd" \
    --priority 4
}

commit
