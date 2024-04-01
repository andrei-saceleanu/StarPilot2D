#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate train_env
python train.py
