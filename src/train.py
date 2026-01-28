"""
@File   :   train.py
@Desc   :   Train inference model of DruG-MAP
"""
from dataset import MorphDataset
from model_all_gan import MorphModel
from utils import *
from torch.utils import data as torch_data
import argparse

logger = get_logger('train', 'sys')
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training MorphGen")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--molecule_path", type=str)
    parser.add_argument("--premodel_path", type=str)
    parser.add_argument("--dev", type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_critic", type=int, default=10)
    parser.add_argument("--loss_weight", type=float, default=1e-3)
    parser.add_argument("--molecule_feature_dim", type=int, default=2304)
    parser.add_argument("--molecule_feature_embed_dim", nargs='+', type=int, default=[1300])
    parser.add_argument("--initialization_model", type=str, default='pretrained_model', help='molecule_feature(pretrained_model, retrain_model)')
    parser.add_argument("--split_data_type", type=str, default='random_split', help='split_data_type(random_split,random_splited, smiles_split, smiles_splited)')
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--pre_epochs", type=int, default=0)
    parser.add_argument("--n_latent", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--train_flag", type=bool, default=False)
    parser.add_argument("--eval_metric", type=bool, default=False)
    parser.add_argument("--predict_profile", type=bool, default=False)
    args = parser.parse_args()
    return args

def train_DruGMAP(args):
    logger.info(args)
    random_seed = args.seed
    setup_seed(random_seed)
    dev = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    # Load profile data
    """
    dict_keys(['CP_index', 'mol_id'.....])
    """
    data = load_HDF5(args.data_path)
    # Extract the cp_index and mol_id columns to split the dataset.
    logger.info(f"==== all data:{len(data['CP_index'])}, {len(data.keys())} ====")
    # Train switch
    train_flag = args.train_flag
    # Evaluation switch
    eval_metric = args.eval_metric
    # Prediction switch
    predict_profile = args.predict_profile
    # If pretrained_model
    init_mode = args.initialization_model
    # Molecule features dim
    features_dim = args.molecule_feature_dim
    # Molecule latent dim
    features_embed_dim = args.molecule_feature_embed_dim
    # Split model
    split_type = args.split_data_type
    # n_folds = 5
    n_folds = 10
    n_epochs = args.n_epochs
    pre_epochs = args.pre_epochs
    premodel_path = args.premodel_path
    # Latent dim
    n_latent = args.n_latent
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    beta = args.beta
    dropout = args.dropout
    weight_decay = args.weight_decay
    n_critic = args.n_critic
    loss_weight = args.loss_weight

    logger.info(f'n_epochs:{n_epochs}, init_mode: {init_mode}, split_type: {split_type}, learning_rate: {learning_rate}, dropout: {dropout}, beta: {beta}, weight decay: {weight_decay}, random_seed: {random_seed}')

    # Out dir
    data_path = '../Dataset/'
    base_out = '../results/trained_models/'
    local_out = base_out + ''
    isExists = os.path.exists(local_out)
    if not isExists:
        os.makedirs(local_out)
        logger.info('==== Out directory created successfully ====')
    else:
        logger.info('==== Out directory already exists ====')

    logger.info('==== split_data ====')
    split_train, split_valid, split_test = split_data(data, n_folds=n_folds, split_type=split_type, rnds=random_seed)

    logger.info(f"==== train {len(set(split_train['CP_index']))}, {len(set(split_train['mol_id']))} ====")
    logger.info(f"==== valid {len(set(split_valid['CP_index']))}, {len(set(split_valid['mol_id']))} ====")
    logger.info(f"==== test {len(set(split_test['CP_index']))}, {len(set(split_test['mol_id']))} ====")

    train = MorphDataset(split_train)
    valid = MorphDataset(split_valid)
    test = MorphDataset(split_test)

    train_loader = torch_data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True, drop_last=False,num_workers=4, worker_init_fn=seed_worker)
    valid_loader = torch_data.DataLoader(dataset=valid, batch_size=batch_size, shuffle=True, drop_last=False,num_workers=4, worker_init_fn=seed_worker)
    test_loader = torch_data.DataLoader(dataset=test, batch_size=batch_size, shuffle=True, drop_last=False,
                                        num_workers=4, worker_init_fn=seed_worker)

    en_hidden = args.en_hidden
    de_hidden = args.de_hidden
    en_hidden_x2 = args.en_hidden_x2
    de_hidden_x2 = args.de_hidden_x2

    if train_flag:
        logger.info('==== Start training ====')

        model = MorphModel(n_features=737,
                           n_latent=n_latent,
                           n_en_hidden=en_hidden,
                           n_de_hidden=de_hidden,
                           n_en_hidden_x2=en_hidden_x2,
                           n_de_hidden_x2=de_hidden_x2,
                           features_dim=features_dim,
                           features_embed_dim=features_embed_dim,
                           init_w=True,
                           beta=beta,
                           device=dev,
                           dropout=dropout,
                           path_model=local_out,
                           random_seed=random_seed,
                           n_critic=n_critic,
                           loss_weight=loss_weight
                           )

        model_dict = model.state_dict()
        # Load different model
        if init_mode == 'pretrained_model':
            logger.info('==== Loading pretrained model ====')

            filename = '../results/pretrain_model_x1/best_model.pt'
            model_base_x1 = torch.load(filename, map_location='cpu')
            model_base_x1_dict = model_base_x1.state_dict()
            for k in model_dict.keys():
                if k in model_base_x1_dict.keys():
                    model_dict[k] = model_base_x1_dict[k]
            filename = '../results/pretrain_model_x2/best_model.pt'
            model_base_x2 = torch.load(filename, map_location='cpu')
            model_base_x2_dict = model_base_x2.state_dict()
            for k in model_dict.keys():
                if k in model_base_x2_dict.keys():
                    model_dict[k] = model_base_x2_dict[k]

            model.load_state_dict(model_dict)
            del model_base_x1, model_base_x2
        elif init_mode == 'retrain_model':
            logger.info('==== Loading last model ====')

            if premodel_path is not None and premodel_path != '':
                filename = premodel_path
            else:
                filename = local_out + 'best_model.pt'
            model_base = torch.load(filename, map_location='cpu')
            model_base_dict = model_base.state_dict()
            for k in model_dict.keys():
                if k in model_base_dict.keys():
                    model_dict[k] = model_base_dict[k]

            model.load_state_dict(model_dict)
            del model_base
            local_out = f'{local_out}retrain/epoch{n_epochs+pre_epochs}/'
            if not os.path.exists(local_out):
                os.makedirs(local_out)
                logger.info('==== local_out directory created successfully ====')
            else:
                logger.info('==== local_out directory already exists ====')
            model.path_model = local_out
        model.to(dev)
        logger.info(model)
        epoch_hist, best_epoch = model.train_model(train_loader=train_loader, test_loader=valid_loader,
                                                   n_epochs=n_epochs, pre_epochs=pre_epochs, learning_rate=learning_rate, weight_decay=weight_decay, save_model=True, logger=logger, metrics_func=['pearson'])

        epoch_result = pd.DataFrame.from_dict(epoch_hist)
        epoch_result['epoch'] = np.arange(n_epochs)
        epoch_result.to_csv(local_out + 'loss_epoch{}.csv'.format(n_epochs),index=False)
        epoch = epoch_result['valid_loss'].idxmin()
        if epoch == best_epoch:
            logger.info(f'==== best valid epoch:{epoch} ====')
        else:
            logger.info(f'==== warning: inconsistent best valid | pearson_best_epoch: {best_epoch}, loss_best_epoch: {epoch} ====')

    filename = local_out + 'best_model.pt'
    model = torch.load(filename, map_location='cpu')
    model.dev = torch.device(dev)
    model.to(dev)

    save_dir = local_out + 'predict'
    isExists = os.path.exists(save_dir)
    logger.info(save_dir)
    if not isExists:
        os.makedirs(save_dir)
        logger.info('==== Predict directory created successfully ====')
    else:
        logger.info('==== Predict directory already exists ====')

    if eval_metric:
        logger.info('==== Evaluate model performance ====')
        setup_seed(random_seed)
        _, _, test_metrics_dict_ls = model.test_model(loader=test_loader, metrics_func=['rmse', 'r2', 'pearson'])

        for name, rec_dict_value in zip(['test'], [test_metrics_dict_ls]):
            df_rec = pd.DataFrame.from_dict(rec_dict_value)
            df_rec.to_csv(save_dir + '/{}_restruction_result.csv'.format(name), index=False)

        logger.info('==== Evaluation completed ====')

if __name__ == "__main__":
    args = parse_args()
    args.en_hidden = [1300]
    args.de_hidden = [1300]
    args.en_hidden_x2 = [1300]
    args.de_hidden_x2 = [1300]
    train_DruGMAP(args)
