if [ -f /sys/hypervisor/uuid ] && [ `head -c 3 /sys/hypervisor/uuid` == ec2 ]; then
    export GLOVE_INDEX_LOC="/home/ubuntu/JeopardyContrastive/exp_vocab/6B.300_idx.pkl"
    export GLOVE_LOC="/home/ubuntu/JeopardyContrastive/exp_vocab/emb_weights_1.data"
    export QUESTIONS_FILE="/home/ubuntu/datasets/OpenEnded_mscoco_train2014_questions.json"
    export ANSWERS_FILE="/home/ubuntu/datasets/mscoco_train2014_annotations.json"
    export COCO_LOC="/home/ubuntu/datasets/train2014"
    export COCO_ROOT="/home/ubuntu/datasets/train2014"
    export EXP_BASE="/home/ubuntu/checkpoints"
else
    export GLOVE_INDEX_LOC="/data5/shashank2000/6B.300_idx.pkl"
    export GLOVE_LOC="/data5/shashank2000/emb_weights_1.data"
    export QUESTIONS_FILE="/data5/shashank2000/final_json/OpenEnded_mscoco_train2014_questions.json"
    export ANSWERS_FILE="/data5/shashank2000/final_json/mscoco_train2014_annotations.json"
    export COCO_LOC="/mnt/fs0/datasets/mscoco/train2014"    
    export EXP_BASE="/mnt/fs5/shashank2000"
fi
