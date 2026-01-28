#!/usr/bin/env bash

python train.py --data_path ../Dataset/Train_harmony_modz_paired.h5 --molecule_path ../Dataset/Train_compounds.pkl --n_epochs 300 --n_latent 100 --batch_size 64 --learning_rate 1e-4 --beta 0.1 --dropout 0.1 --weight_decay 1e-5 --n_critic 1 --loss_weight 1e-3 --train_flag True --eval_metric True --seed 42 --dev cuda:0