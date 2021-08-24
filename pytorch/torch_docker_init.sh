sudo nvidia-docker run \
--name pytorch_trial \
-it \
--shm-size 16G \
-v /data/crz/cs-study-on-linux/pytorch/ML-course-LHY:/myworkspace \
nvcr.io/nvidia/pytorch:20.03-py3 \
/bin/bash
