#!/bin/bash
# hs: ws1lu4iyv5yjjyvp
# ddd: ws1ujefpjyfgqjwp

# zouyicheng data source data1ewbw1ztmmyh
# doushihan data source data1dfp0cngxv41
# liushichun1 data source data1ubhj4714msc
# geqiming data1xj7ojru0t4t

export HOME="/cpfs01/shared/llm_ddd/zouyicheng/"


function commit {
    num_nodes=1
    name="xtuner-convert-pth-to-hf"
    num_tasks_per_node=1
    node_cpus=96
    num_gpus=1
    node_mems=512Gi

    cmd=". /cpfs01/shared/llm_ddd/zouyicheng/.bashrc && conda activate xtuner && \
cd /cpfs01/shared/llm_ddd/zouyicheng/xtuner && PYTHONPATH=. python /cpfs01/shared/llm_ddd/zouyicheng/xtuner/xtuner/tools/model_converters/pth_to_hf.py \
/cpfs01/shared/llm_ddd/zouyicheng/xtuner/work_dirs/internlm2_5_1_8b_reward_9_9m_single_mix_test/internlm2_5_1_8b_reward_9_9m_single_mix_test.py \
/cpfs01/shared/llm_ddd/zouyicheng/xtuner/work_dirs/internlm2_5_1_8b_reward_9_9m_single_mix_test/iter_130.pth  \
/cpfs01/shared/llm_ddd/zouyicheng/xtuner/work_dirs/internlm2_5_1_8b_reward_9_9m_single_mix_test/iter_130_hf"

    /cpfs01/shared/public/dlc create job --config /cpfs01/shared/public/zouyicheng/dlc.config \
    --name $name \
    --worker_count $num_nodes \
    --kind PyTorchJob \
    --worker_cpu $node_cpus \
    --worker_gpu $num_gpus \
    --worker_memory $node_mems \
    --workspace_id ws1lu4iyv5yjjyvp \
    --data_sources data1ewbw1ztmmyh,data1bgvj0n14to0,data1dfp0cngxv41,data1ubhj4714msc,data1xj7ojru0t4t,data9qbxtzsqaa1f   \
    --worker_image pjlab-shanghai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/pjlab-eflops/lishuaibin:lishuaibin-xpuyu-trainrlhf \
    --worker_shared_memory 256Gi \
    --command "$cmd" \
    --priority 4
}

commit 
