"""
@File   :   predict.py
@Desc   :   predicting MOA with trained DruG-MAP classification model
"""
from utils import *
import argparse
import warnings
import xgboost as xgb
warnings.filterwarnings('ignore')

logger = get_logger('predict', 'sys')
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for prediction")
    parser.add_argument("--profiles_path", type=str, default='../prediction/results/prediction_profiles.h5')
    parser.add_argument("--out_path", type=str, default='../prediction/results/')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args

# predicting MOA
def classifier_predict(args):
    random_seed = args.seed
    setup_seed(random_seed)
    out_path = args.out_path
    profiles_path = args.profiles_path
    model_xgb = xgb.XGBClassifier()
    model_xgb.load_model('../prediction/model/classifier/xgboost.model')
    pred_data = load_HDF5(profiles_path)

    deg_pred = pred_data['x2_pred'] - pred_data['x1']
    y_pred_drug = model_xgb.predict_proba(deg_pred)
    # Output model report and view evaluation metrics
    labels = pd.read_csv('../prediction/model/classifier/MOAs.csv')
    df = {}
    for idx, moa in enumerate(labels['MOAs']):
        if idx == 0:
            df['MOA'] = [moa]
            for index, cp_id in enumerate(pred_data['inchi']):
                df[cp_id] = [round(y_pred_drug[index][idx], 4)]
        else:
            df['MOA'] += [moa]
            for index, cp_id in enumerate(pred_data['inchi']):
                df[cp_id] += [round(y_pred_drug[index][idx], 4)]

    df = pd.DataFrame(df)
    # df = df.T
    df.to_csv(f'{out_path}/prediction_moa.csv', index=False)
    print("Predicting MOA complete!")

if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    classifier_predict(args)