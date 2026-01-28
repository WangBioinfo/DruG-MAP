#!/usr/bin/env bash

python train_x2.py --data_path ../Dataset/Train_harmony_modz_paired.h5 --n_epochs 300 --n_latent 100 --batch_size 64 --learning_rate 1e-3 --beta 0.1 --dropout 0.1 --weight_decay 1e-5 --train_flag True --eval_metric True --seed 42 --n_critic 10 --dev cuda:0
