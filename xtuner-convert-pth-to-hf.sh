#!/bin/bash
# hs: ws1lu4iyv5yjjyvp
# ddd: ws1ujefpjyfgqjwp

# zouyicheng data source data1ewbw1ztmmyh
# doushihan data source data1dfp0cngxv41
# liushichun1 data source data1ubhj4714msc
# geqiming data1xj7ojru0t4t

export HOME="/cpfs01/shared/alillm_hs/zouyicheng/"


function commit {
    num_nodes=1
    name="xtuner-convert-pth-to-hf"
    num_tasks_per_node=1
    node_cpus=16
    num_gpus=1
    node_mems=128Gi

    cmd=". /cpfs01/shared/alillm_hs/zouyicheng/.bashrc && conda activate xtuner && \
cd /cpfs01/shared/alillm_hs/zouyicheng/xtuner && PYTHONPATH=. python /cpfs01/shared/alillm_hs/zouyicheng/xtuner/xtuner/tools/model_converters/pth_to_hf.py \
/cpfs01/shared/alillm_hs/zouyicheng/xtuner/work_dirs/RM_PT_internlm2_5_7b_DATA_510m_single_mix_Node_57_LR_1_45e_5/RM_PT_internlm2_5_7b_DATA_510m_single_mix_Node_57_LR_1_45e_5.py \
/cpfs01/shared/alillm_hs/zouyicheng/xtuner/work_dirs/RM_PT_internlm2_5_7b_DATA_510m_single_mix_Node_57_LR_1_45e_5/iter_21000.pth  \
/cpfs01/shared/alillm_hs/zouyicheng/xtuner/work_dirs/RM_PT_internlm2_5_7b_DATA_510m_single_mix_Node_57_LR_1_45e_5/iter_21000_hf"

    /cpfs01/shared/public/dlc create job --config /cpfs01/shared/public/zouyicheng/dlc.config \
    --name $name \
    --worker_count $num_nodes \
    --kind PyTorchJob \
    --worker_cpu $node_cpus \
    --worker_gpu $num_gpus \
    --worker_memory $node_mems \
    --workspace_id ws1ujefpjyfgqjwp \
    --data_sources data1ewbw1ztmmyh,data1bgvj0n14to0,data1dfp0cngxv41,data1ubhj4714msc,data1xj7ojru0t4t,data9qbxtzsqaa1f,data1o8qdjce0kd0  \
    --worker_image pjlab-shanghai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/pjlab-eflops/lishuaibin:lishuaibin-xpuyu-trainrlhf \
    --worker_shared_memory 256Gi \
    --command "$cmd" \
    --priority 4
}

commit 
