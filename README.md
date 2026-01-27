# **DruG-MAP** 

## **Introduction**
DruG-MAP (Deep Generative Model for Drug-induced Morphological Profile Analysis and Prediction), a deep generative framework based on the concept of virtual cells, establishing a complete research workflow from compound structure to morphological profile prediction, drug function inference, and experimental validation.

## **Predict**
### Inferring morphological profile
Inferring the morphological profile of target compounds
```
> ./infer.sh
```
or
```
python infer.py  --model_path ../prediction/model/inference/infer_model_best.pt --molecule_path ../prediction/data/prediction_compounds.csv --molecule_feature_path ../prediction/data/prediction_compounds.pkl --seed 42
```
### Predicting MOA via inferred morphological profiles
```
> ./predict.sh
```
or
```
python predict.py  --profiles_path ../prediction/results/prediction_profiles.h5
```

## **Train**
### Reconstruction model
Pretraining reconstruction model for control (x1) and treantment (x2) profile

Training can be started via script
```
> ./train_x1.sh

> ./train_x2.sh
```
or run the Python script manually
```
python train_x1.py --data_path ../Dataset/Train_harmony_modz_paired.h5 --n_epochs 300 --seed 42

python train_x2.py --data_path ../Dataset/Train_harmony_modz_paired.h5 --n_epochs 300 --seed 42
```

### Inference model
Training can be started via script
```
> ./train.sh
```
or run the Python script manually
```
python train.py --data_path ../Dataset/Train_harmony_modz_paired.h5 --molecule_path ../Dataset/Train_compounds.pkl --n_epochs 300 --seed 42
```
## **Requirements**
python = 3.8.19

pytorch = 2.2.0

pandas = 2.0.3

numpy = 1.24.1

scikit-learn = 1.3.2

rdkit = 2024.3.2

xgboost = 2.1.1

All dependencies are in requirements.txt.