#!/bin/bash
LOCAL_HOSTNAME=$(hostname -d)
if [[ ${LOCAL_HOSTNAME} =~ .*\.amazonaws\.com ]]
then
        echo "This is an EC2 instance"
        export COCO_LOC=""
        export CHECKPOINT_LOC=""
else
        echo "This is not an EC2 instance, or a reverse-customized one"
        export COCO_LOC=""
        export CHECKPOINT_LOC=""
fi
