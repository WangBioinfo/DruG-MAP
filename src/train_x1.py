"""
@File   :   train_x1
@Desc   :   Pretrain x1 model
"""

from dataset import *
from model_x1 import Model_x1
from utils import *
import argparse
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dev", type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--n_latent", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--train_flag", type=bool, default=True)
    parser.add_argument("--eval_metric", type=bool, default=True)
    parser.add_argument("--n_critic", type=int, default=10)
    args = parser.parse_args()
    return args

def train_x1(args):
    print(args)
    # Get random seed
    random_seed = args.seed
    setup_seed(random_seed)
    dev = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    # Load data
    """
    dict_keys(['CP_index', 'mol_id'.....])
    """
    data = load_HDF5(args.data_path)
    print('all data:', len(data['CP_index']), len(data.keys()))
    train_flag = args.train_flag
    eval_metric = args.eval_metric
    n_folds = 5
    n_epochs = args.n_epochs
    n_latent = args.n_latent
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    beta = args.beta
    dropout = args.dropout
    weight_decay = args.weight_decay
    n_critic = args.n_critic

    print('train_flag:', train_flag, 'eval_metric:', eval_metric, 'learning_rate:', learning_rate, 'dropout:', dropout, 'beta:', beta, 'weight decay:', weight_decay, 'random_seed:', random_seed)

    # Out dir
    local_out = '../results/pretrain_model_x1/'
    isExists = os.path.exists(local_out)
    if not isExists:
        os.makedirs(local_out)
        print('Directory created successfully')
    else:
        print('Directory already exists')
    # Split dataset
    # 'random_split', 'smiles_split'
    pair, pairv, pairt = split_data(data, n_folds=n_folds, split_type='random_split', rnds=random_seed)
    print('train', len(set(pair['CP_index'])), len(pair['mol_id']))
    print('valid', len(set(pairv['CP_index'])), len(pairv['mol_id']))
    print('test', len(set(pairt['CP_index'])), len(pairt['mol_id']))

    train = Dataset_x1(pair)
    valid = Dataset_x1(pairv)
    test = Dataset_x1(pairt)

    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, worker_init_fn=seed_worker)
    valid_loader = torch.utils.data.DataLoader(dataset=valid, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, worker_init_fn=seed_worker)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, worker_init_fn=seed_worker)

    en_hidden = args.en_hidden
    de_hidden = args.de_hidden
    if train_flag:
        model = Model_x1(n_features=737, n_latent=n_latent, n_en_hidden=en_hidden, n_de_hidden=de_hidden,
                       init_w=True, beta=beta, device=dev, dropout=dropout,
                       path_model=local_out, random_seed=random_seed, BCE_pos_weight=None, n_critic=n_critic)
        model.to(dev)
        print(model)
        # Training
        epoch_hist, best_epoch = model.train_model(train_loader=train_loader, test_loader=valid_loader,
                                                   n_epochs=n_epochs, learning_rate=learning_rate,
                                                   weight_decay=weight_decay, save_model=True)

        epoch_result = pd.DataFrame.from_dict(epoch_hist)
        epoch_result['epoch'] = np.arange(n_epochs)
        # Save loss value for each epoch
        epoch_result.to_csv(local_out + 'loss_epoch{}.csv'.format(n_epochs), index=False)
        # idxmin: Return the row label of the minimum value.
        epoch = epoch_result['valid_loss'].idxmin()
        if epoch == best_epoch:
            print('best valid epoch:', epoch)
        else:
            print('warning: inconsistent best valid')

    filename = local_out + 'best_model.pt'
    model = torch.load(filename, map_location='cpu')
    model.dev = torch.device(dev)
    model.to(dev)

    # Evaluation
    if eval_metric:
        print('===============Evaluate model performance==============')
        save_dir = local_out + 'predict'
        isExists = os.path.exists(save_dir)
        print(save_dir)
        if not isExists:
            os.makedirs(save_dir)
            print('Directory created successfully')
        else:
            print('Directory already exists')

        setup_seed(random_seed)
        _, _, test_metrics_dict_ls = model.test_model(loader=test_loader, metrics_func=['pearson', 'r2', 'rmse'])

        for name, rec_dict_value in zip(['test'], [test_metrics_dict_ls]):
            df_rec = pd.DataFrame.from_dict(rec_dict_value)
            df_rec.to_csv(save_dir + '/{}_restruction_eval_x1_epoch{}.csv'.format(name, n_epochs), index=False)
        ddict_data_test = model.recon_profile(loader=test_loader)
        save_HDF5(save_dir + '/test_restruction_profile.h5', ddict_data_test)
        print('Evaluation of the pretrain dataset x1 is complete.')

if __name__ == "__main__":
    args = parse_args()
    args.en_hidden = [1300]
    args.de_hidden = [1300]
    train_x1(args)
