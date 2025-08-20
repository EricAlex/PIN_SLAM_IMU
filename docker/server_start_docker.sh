#!/bin/bash

docker run -it --rm --ipc host -p 8080:8081 --gpus all \
-v /home/xin/Downloads:/storage/  \
pinslam:localbuild \
