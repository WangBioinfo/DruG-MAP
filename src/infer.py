"""
@File   :   infer.py
@Desc   :   Inferring profile with trained DruG-MAP inference model
"""
import numpy as np
from torch.utils import data as torch_data
from dataset import MorphDataset_prediction, MorphDataset
from utils import *
import pickle
import argparse
import warnings
import xgboost as xgb
warnings.filterwarnings('ignore')

logger = get_logger('infer', 'sys')
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for prediction")
    parser.add_argument("--model_path", type=str, default='../prediction/model/inference/infer_model_best.pt')
    parser.add_argument("--molecule_path", type=str, default='../prediction/data/prediction_compounds.csv')
    parser.add_argument("--molecule_feature_path", type=str, default='../prediction/data/prediction_compounds.pkl')
    parser.add_argument("--out_path", type=str, default='../prediction/results/')
    parser.add_argument("--x1_path", type=str, default='../prediction/data/x1.h5')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev", type=str, default='cuda:0')

    args = parser.parse_args()
    return args

# prediction
def prediction_profiles(args):
    logger.info(args)
    random_seed = args.seed
    setup_seed(random_seed)
    dev = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path
    molecule_path = args.molecule_path
    molecule_feature_path = args.molecule_feature_path
    out_path = args.out_path
    x1_data = load_HDF5(args.x1_path)
    mol_ids = pd.read_csv(molecule_path)
    with open(molecule_feature_path, 'rb') as f:
        mol_features = pickle.load(f)

    model = torch.load(model_path, map_location='cpu')
    model.dev = torch.device(dev)
    model.to(dev)
    # Use "Smiles" as the key for regular compound PKL files
    x1_dataset = MorphDataset_prediction(x1_data['x1'], mol_features, mol_ids['Smiles'])
    loader = torch_data.DataLoader(dataset=x1_dataset, batch_size=64, shuffle=True, drop_last=False, num_workers=0, worker_init_fn=seed_worker)
    logger.info('===============Predict profile==============')
    x2_pred_array, mol_id_array, z2_pred_array = model.predict_profile_for_x1(loader)
    ddict_data = dict()
    ddict_data['x1'] = x1_dataset.x1
    ddict_data['x2_pred'] = x2_pred_array
    ddict_data['mol_id'] = mol_id_array
    ddict_data['z2_pred'] = z2_pred_array

    for k, mol in enumerate(mol_id_array):
        # Use "Smiles" as the key for regular compound PKL files
        mol_data = mol_ids[mol_ids['Smiles'] == mol]
        for k2 in mol_data:
            try:
                ddict_data[k2] = np.concatenate((ddict_data[k2], mol_data[k2]), axis=0)
            except:
                ddict_data[k2] = np.array(mol_data[k2]).astype(np.str_)

    for k in ddict_data.keys():
        logger.info(f'{type(ddict_data[k][0])}, {ddict_data[k].shape}')

    check_path_exists(out_path)
    save_HDF5(out_path + 'prediction_profiles.h5', ddict_data)
    print("Inferring morphological profiles complete!")

if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    prediction_profiles(args)