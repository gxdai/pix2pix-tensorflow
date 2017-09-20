#!/bin/bash

# $1:   GPU index
# $2:   train or test


CUDA_VISIBLE_DEVICES=$1 python main.py --phase $2 --dataset_name medicalImage \
    --input_nc 1 --output_nc 1 --imageRootDir "../../data/medicalImage/wustl/TrainingSet"
