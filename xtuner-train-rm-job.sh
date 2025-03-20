nodes=8

config="RM_PT_internlm2_5_1_8b_DATA_9_9m_single_mix_Node_8_LR_1e_5"


TARGET_FILE=/cpfs01/shared/llm_ddd/zouyicheng/xtuner/addr/addr_${name}.txt
RANK=${RANK:-0}
MASTER_PORT=6382
MASTER_ADDR=${MASTER_ADDR}
echo "MASTER_ADDR: $MASTER_ADDR"



echo "Rank $RANK is running on $MASTER_ADDR"
if [ "$RANK" -eq 0 ]; then 
    echo "Starting head node (RANK=${RANK}) on port $MASTER_PORT..."
    
    MASTER_ADDR=${MASTER_ADDR}
    echo "$MASTER_ADDR" > "$TARGET_FILE"

    sleep 20
    
    NPROC_PER_NODE=8 NNODES=$nodes PORT=$MASTER_PORT ADDR=$MASTER_ADDR NODE_RANK=$RANK xtuner train /cpfs01/shared/llm_ddd/zouyicheng/xtuner/configs/$config.py --deepspeed deepspeed_zero1 
    
else 
    sleep 30
    MASTER_ADDR=$(cat "$TARGET_FILE")

    echo "Starting worker node (RANK=${RANK}), connecting to ${MASTER_ADDR}:${MASTER_PORT}."
    NPROC_PER_NODE=8 NNODES=$nodes PORT=$MASTER_PORT ADDR=$MASTER_ADDR NODE_RANK=$RANK xtuner train /cpfs01/shared/llm_ddd/zouyicheng/xtuner/configs/$config.py --deepspeed deepspeed_zero1 
fi
