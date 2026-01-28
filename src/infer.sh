#!/usr/bin/env bash

python prediction.py --model_path ../prediction/model/inference/infer_model_best.pt --molecule_path ../prediction/data/prediction_compounds.csv --seed 42 --dev cuda:0
