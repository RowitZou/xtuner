nodes=2

config="RM_SFT_internlm2_5_7b_DATA_HH_puyu_mixed_Node_2_LR_1e_5"


TARGET_FILE=/cpfs01/shared/alillm_hs/zouyicheng/xtuner/addr/addr_${config}.txt
RANK=${RANK:-0}
MASTER_PORT=6382
MASTER_ADDR=${MASTER_ADDR}
echo "MASTER_ADDR: $MASTER_ADDR"



echo "Rank $RANK is running on $MASTER_ADDR"
if [ "$RANK" -eq 0 ]; then 
    echo "Starting head node (RANK=${RANK}) on port $MASTER_PORT..."
    
    MASTER_ADDR=${MASTER_ADDR}
    echo "$MASTER_ADDR" > "$TARGET_FILE"

    sleep 30
    
    NPROC_PER_NODE=8 NNODES=$nodes PORT=$MASTER_PORT ADDR=$MASTER_ADDR NODE_RANK=$RANK xtuner train /cpfs01/shared/alillm_hs/zouyicheng/xtuner/configs/$config.py --deepspeed deepspeed_zero1 
    
else
    sleep 70
    MASTER_ADDR=$(cat "$TARGET_FILE")

    echo "Starting worker node (RANK=${RANK}), connecting to ${MASTER_ADDR}:${MASTER_PORT}."
    NPROC_PER_NODE=8 NNODES=$nodes PORT=$MASTER_PORT ADDR=$MASTER_ADDR NODE_RANK=$RANK xtuner train /cpfs01/shared/alillm_hs/zouyicheng/xtuner/configs/$config.py --deepspeed deepspeed_zero1 
fi
