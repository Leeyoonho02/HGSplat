# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

scene='./data/IMG_4190'
images='images'

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

ulimit -n 4096
port=$(rand 10000 30000)

timestamp=$(date "+%Y-%m-%d_%H:%M:%S")

python train.py --eval -s ${scene} -m outputs/${scene}/${timestamp} --port ${port} --images ${images} --mode custom -r 4 --pose_iteration 100 --local_iter 200 --global_iter 500
python render.py -m outputs/${scene}/${timestamp}
python metrics.py -m outputs/${scene}/${timestamp}