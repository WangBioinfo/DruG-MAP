"""
@File   :   model.py
@Desc   :   Inference model of DruG-MAP, x1: Control, x2: Treatment
"""
import torch.nn.functional as F
from torch import nn, optim
from collections import defaultdict

from utils import *
from model_x2 import Discriminator, Model_x2
from model_x1 import Model_x1

# Define the DruGMAP object
class MorphModel(torch.nn.Module):
    # Constructor for class DrugMAP
    def __init__(self, n_features, n_latent, n_en_hidden, n_de_hidden, features_dim, features_embed_dim, **kwargs):
        super(MorphModel, self).__init__()
        self.n_features = n_features
        self.n_latent = n_latent
        self.n_en_hidden = n_en_hidden
        self.n_de_hidden = n_de_hidden
        self.n_en_hidden_x2 = kwargs.get('n_en_hidden_x2', [1300])
        self.n_de_hidden_x2 = kwargs.get('n_de_hidden_x2', [1300])
        self.features_dim = features_dim
        self.feat_embed_dim = features_embed_dim
        self.init_w = kwargs.get('init_w', False)
        self.beta = kwargs.get('beta', 0.05)
        self.path_model = kwargs.get('path_model', 'trained_model')
        self.dev = kwargs.get('device', torch.device('cpu'))
        self.dropout = kwargs.get('dropout', 0.3)
        self.random_seed = kwargs.get('random_seed', 20240729)
        self.n_critic = kwargs.get('n_critic')
        self.loss_weight = kwargs.get('loss_weight', 1e-3)

        model_x1 = Model_x1(n_features=737, n_latent=n_latent, n_en_hidden=self.n_en_hidden, n_de_hidden=self.n_de_hidden,init_w=True, beta=self.beta, device=self.dev, dropout=self.dropout,
                         path_model=self.path_model, random_seed=self.random_seed, BCE_pos_weight=None)
        self.encoder_x1 = model_x1.encoder_x1
        self.mu_z1 = model_x1.mu_z1
        self.logvar_z1 = model_x1.logvar_z1
        self.decoder_x1 = model_x1.decoder_x1

        model_x2 = Model_x2(n_features=737, n_latent=n_latent, n_en_hidden=self.n_en_hidden_x2, n_de_hidden=self.n_de_hidden_x2,init_w=True, beta=self.beta, device=self.dev, dropout=self.dropout,
                         path_model=self.path_model, random_seed=self.random_seed, BCE_pos_weight=None, n_critic=self.n_critic)

        self.encoder_x2 = model_x2.encoder_x2
        self.mu_z2 = model_x2.mu_z2
        self.logvar_z2 = model_x2.logvar_z2
        self.decoder_x2 = model_x2.decoder_x2

        if self.feat_embed_dim == None:
            self.mu_z2Fz1 = nn.Sequential(nn.Linear(self.n_latent + self.features_dim, self.n_latent), )
            self.logvar_z2Fz1 = nn.Sequential(nn.Linear(self.n_latent + self.features_dim, self.n_latent), )
        else:
            feat_embeddings = [
                nn.Linear(self.features_dim, self.feat_embed_dim[0]),
                nn.BatchNorm1d(self.feat_embed_dim[0]),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout)
            ]
            if len(self.feat_embed_dim) > 1:
                for i in range(len(self.feat_embed_dim) - 1):
                    feat_hidden = [
                        nn.Linear(self.feat_embed_dim[i], self.feat_embed_dim[i + 1]),
                        nn.BatchNorm1d(self.feat_embed_dim[i + 1]),
                        nn.LeakyReLU(),
                        nn.Dropout(self.dropout)
                    ]
                    feat_embeddings = feat_embeddings + feat_hidden
            self.feat_embeddings = nn.Sequential(*feat_embeddings)

            self.mu_z2Fz1 = nn.Sequential(nn.Linear(self.n_latent + self.feat_embed_dim[-1], self.n_latent), )
            self.logvar_z2Fz1 = nn.Sequential(nn.Linear(self.n_latent + self.feat_embed_dim[-1], self.n_latent), )

        if self.init_w:
            self.encoder_x1.apply(self._init_weights)
            self.decoder_x1.apply(self._init_weights)

            self.encoder_x2.apply(self._init_weights)
            self.decoder_x2.apply(self._init_weights)

            self.mu_z1.apply(self._init_weights)
            self.logvar_z1.apply(self._init_weights)

            self.mu_z2.apply(self._init_weights)
            self.logvar_z2.apply(self._init_weights)

    def _init_weights(self, layer):
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
        return

    def encode_x1(self, x1):
        y = self.encoder_x1(x1)
        mu1, logvar1 = self.mu_z1(y), self.logvar_z1(y)
        z1 = self.sample_latent(mu1, logvar1)
        return z1, mu1, logvar1

    def encode_x2(self, x2):
        y = self.encoder_x2(x2)
        mu2, logvar2 = self.mu_z2(y), self.logvar_z2(y)
        z2 = self.sample_latent(mu2, logvar2)
        return z2, mu2, logvar2

    def decode_x1(self, z1):
        x1_rec = self.decoder_x1(z1)
        return x1_rec

    def decode_x2(self, z2):
        x2_rec = self.decoder_x2(z2)
        return x2_rec

    def sample_latent(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.dev)
        eps = eps.mul_(std).add_(mu)
        return eps

    def forward(self, x1, features):
        z1, mu1, logvar1 = self.encode_x1(x1)
        x1_rec = self.decode_x1(z1)
        if self.feat_embed_dim != None:
            feat_embed = self.feat_embeddings(features)
        else:
            feat_embed = features
        # Concat the encoded x1->z1 with the compound features
        z1_feat = torch.cat([z1, feat_embed], 1)
        mu_pred, logvar_pred = self.mu_z2Fz1(z1_feat), self.logvar_z2Fz1(z1_feat)
        # z2Fz1: Latent and perturbation representations from x1
        z2_pred = self.sample_latent(mu_pred, logvar_pred)
        # Decoding z2Fz1 yields x2', the predicted profile.
        x2_pred = self.decode_x2(z2_pred)
        return x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred

    def loss(self, x1_train, x1_rec, mu1, logvar1, x2_train, x2_rec, mu2, logvar2, x2_pred, mu_pred, logvar_pred):
        mse_x1 = F.mse_loss(x1_rec, x1_train, reduction="sum")
        mse_x2 = F.mse_loss(x2_rec, x2_train, reduction="sum")
        mse_x2_pred = F.mse_loss(x2_pred, x2_train, reduction="sum")
        mse_pert = F.mse_loss(x2_pred - x1_train, x2_train - x1_train, reduction="sum")

        kld_x1 = -0.5 * torch.sum(1. + logvar1 - mu1.pow(2) - logvar1.exp(), )
        kld_x2 = -0.5 * torch.sum(1. + logvar2 - mu2.pow(2) - logvar2.exp(), )
        kld_x2_pred = -0.5 * torch.sum(1. + logvar_pred - mu_pred.pow(2) - logvar_pred.exp(), )
        kld_pert = -0.5 * torch.sum(
            1 + (logvar_pred - logvar2) - ((mu_pred - mu2).pow(2) + logvar_pred.exp()) / logvar2.exp(), )

        loss = mse_x2_pred + mse_pert + self.beta * kld_x2_pred + self.beta * kld_pert

        return loss, mse_x1, mse_x2, mse_pert, kld_x1, kld_x2, kld_pert

    def train_model(self, learning_rate, weight_decay, n_epochs, train_loader, test_loader, logger, pre_epochs, save_model=True, metrics_func=None):
        D = Discriminator(n_features=self.n_features, dropout=self.dropout, dev=self.dev)
        D.to(self.dev)

        epoch_hist = defaultdict(list)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
        optimizer_disc = optim.Adam(D.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))

        # Freeze the parameters of encoder_x1 layer
        # for name, param in self.named_parameters():
        #     if "encoder_x1" in name or "mu_z1" in name or "logvar_z1" in name:
        #         param.requires_grad = False

        loss_item = ['loss', 'mse_x1', 'mse_x2', 'mse_pert', 'kld_x1', 'kld_x2', 'kld_pert', 'loss_D', 'loss_G']

        # Train
        best_value = np.inf
        best_pearson = 0
        best_epoch = 0
        for epoch in range(pre_epochs, pre_epochs+n_epochs):
            train_size = 0
            loss_value = 0
            self.train()
            for i, (x1_train, x2_train, features, mol_id, CP_index) in enumerate(train_loader):
                x1_train = x1_train.to(self.dev)
                x2_train = x2_train.to(self.dev)
                features = features.to(self.dev)
                if x1_train.shape[0] == 1:
                    continue
                train_size += x1_train.shape[0]

                x1_rec, mu1, logvar1, x2_pert, mu_pred, logvar_pred, z2_pred = self.forward(x1_train, features)
                z2, mu2, logvar2 = self.encode_x2(x2_train)
                x2_rec = self.decode_x2(z2)

                loss, _, _, _, _, _, _ = self.loss(x1_train, x1_rec, mu1, logvar1, x2_train, x2_rec, mu2, logvar2,
                                                   x2_pert, mu_pred, logvar_pred)
                n_critic = self.n_critic
                for _ in range(n_critic):
                    # Train discriminator
                    optimizer_disc.zero_grad()
                    outputs_real = D.forward(x2_train)
                    outputs_fake = D.forward(x2_rec.detach())

                    lambda_gp = 10
                    loss_disc = D.dis_loss(x2_train, x2_rec.detach(), outputs_real, outputs_fake, lambda_gp)

                    loss_disc.backward()
                    optimizer_disc.step()

                outputs_fake_gen = D.forward(x2_rec)
                loss_gen = D.gen_loss(outputs_fake_gen)
                optimizer.zero_grad()
                # Total loss
                loss_vae_gan = loss_gen + self.loss_weight * loss
                loss_value += loss.item()
                loss_vae_gan.backward()
                optimizer.step()

            # Eval
            train_dict, train_metrics_dict, train_metrics_dict_ls = self.test_model(loader=train_loader,
                                                                                    loss_item=loss_item,
                                                                                    metrics_func=None)
            train_loss = train_dict['loss']
            train_mse_x1 = train_dict['mse_x1']
            train_mse_x2 = train_dict['mse_x2']
            train_mse_pert = train_dict['mse_pert']
            train_kld_x1 = train_dict['kld_x1']
            train_kld_x2 = train_dict['kld_x2']
            train_kld_pert = train_dict['kld_pert']
            train_loss_D = train_dict['loss_D']
            train_loss_G = train_dict['loss_G']

            for k, v in train_dict.items():
                epoch_hist['train_' + k].append(v)

            for k, v in train_metrics_dict.items():
                epoch_hist['train_' + k].append(v)

            test_dict, test_metrics_dict, test_metricsdict_ls = self.test_model(loader=test_loader, loss_item=loss_item,
                                                                                metrics_func=metrics_func)
            test_loss = test_dict['loss']
            test_mse_x1 = test_dict['mse_x1']
            test_mse_x2 = test_dict['mse_x2']
            test_mse_pert = test_dict['mse_pert']
            test_kld_x1 = test_dict['kld_x1']
            test_kld_x2 = test_dict['kld_x2']
            test_kld_pert = test_dict['kld_pert']
            test_loss_D = train_dict['loss_D']
            test_loss_G = train_dict['loss_G']
            test_pearson = test_metrics_dict['x2_pred_pearson']
            test_deg_pearson = test_metrics_dict['DEG_pred_pearson']

            for k, v in test_dict.items():
                epoch_hist['valid_' + k].append(v)
            for k, v in test_metrics_dict.items():
                epoch_hist['valid_' + k].append(v)

            logger.info(
                f'[Epoch {epoch}] | loss: {train_loss:.3f}, train_loss_D: {train_loss_D:.5f}, train_loss_G: {train_loss_G:.5f}| valid_loss: {test_loss:.3f}, valid_loss_D: {test_loss_D:.5f}, valid_loss_G: {test_loss_G:.5f} | last_loss: {loss:.3f}, last_loss_all: {loss_vae_gan:.5f}, last_loss_G: {loss_gen:.5f}, last_loss_D: {loss_disc:.5f} | test_pearson: {test_pearson:.3f}, test_deg_pearson: {test_deg_pearson:.3f}')

            #if test_loss < best_value:
                #best_value = test_loss
            if test_deg_pearson > best_pearson:
                best_pearson = test_deg_pearson
                best_epoch = epoch
                if save_model:
                    torch.save(self, self.path_model + 'best_model.pt')

        return epoch_hist, best_epoch

    def test_model(self, loader, loss_item=None, metrics_func=None):
        D = Discriminator(n_features=self.n_features, dropout=self.dropout, dev=self.dev)
        D.to(self.dev)

        test_dict = defaultdict(float)
        metrics_dict_all = defaultdict(float)
        metrics_dict_all_ls = defaultdict(list)
        test_size = 0

        self.eval()
        with torch.no_grad():
            for x1_data, x2_data, mol_features, mol_id, CP_index in loader:
                x1_data = x1_data.to(self.dev)
                x2_data = x2_data.to(self.dev)
                mol_features = mol_features.to(self.dev)
                CP_index = np.array(list(CP_index))
                mol_id = np.array(list(mol_id))
                test_size += x1_data.shape[0]

                x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = self.forward(x1_data, mol_features)
                z2, mu2, logvar2 = self.encode_x2(x2_data)
                x2_rec = self.decode_x2(z2)
                loss_ls = self.loss(x1_data, x1_rec, mu1, logvar1, x2_data, x2_rec, mu2, logvar2, x2_pred, mu_pred,
                                    logvar_pred)

                outputs_real = D.forward(x2_data)
                outputs_fake = D.forward(x2_rec.detach())
                loss_D = -D.wasserstein_loss(outputs_real) + D.wasserstein_loss(outputs_fake)
                outputs_fake_gen = D.forward(x2_rec)
                loss_G = D.gen_loss(outputs_fake_gen)

                loss_ls = list(loss_ls)
                loss_ls.append(loss_D)
                loss_ls.append(loss_G)

                if loss_item is not None:
                    for idx, k in enumerate(loss_item):
                        test_dict[k] += loss_ls[idx].item()

                if metrics_func is not None:
                    metrics_dict, metrics_dict_ls = self.eval_x_reconstruction(x1_data, x1_rec, x2_data, x2_rec,
                                                                               x2_pred, metrics_func=metrics_func)
                    for k in metrics_dict.keys():
                        metrics_dict_all[k] += metrics_dict[k]
                    for k in metrics_dict_ls.keys():
                        metrics_dict_all_ls[k] += metrics_dict_ls[k]

                    metrics_dict_all_ls['mol_id'] += list(mol_id)
                    metrics_dict_all_ls['CP_index'] += list(CP_index)

                try:
                    x1_array = torch.cat([x1_array, x1_data], dim=0)
                    x1_rec_array = torch.cat([x1_rec_array, x1_rec], dim=0)
                    x2_array = torch.cat([x2_array, x2_data], dim=0)
                    x2_rec_array = torch.cat([x2_rec_array, x2_rec], dim=0)
                    x2_pred_array = torch.cat([x2_pred_array, x2_pred], dim=0)
                    mol_id_array = np.concatenate((mol_id_array, mol_id), axis=0)
                    cp_index_array = np.concatenate((cp_index_array, CP_index), axis=0)
                except:
                    x1_array = x1_data.clone()
                    x1_rec_array = x1_rec.clone()
                    x2_array = x2_data.clone()
                    x2_rec_array = x2_rec.clone()
                    x2_pred_array = x2_pred.clone()
                    mol_id_array = mol_id.copy()
                    cp_index_array = CP_index.copy()

        for k in test_dict.keys():
            test_dict[k] = test_dict[k] / test_size

        for k in metrics_dict_all.keys():
            metrics_dict_all[k] = metrics_dict_all[k] / test_size

        return test_dict, metrics_dict_all, metrics_dict_all_ls

    def predict_profile(self, loader):
        test_size = 0
        self.eval()
        with torch.no_grad():
            for x1_data, x2_data, mol_features, mol_id, CP_index in loader:
                CP_index = np.array(list(CP_index))
                mol_id = np.array(list(mol_id))
                x1_data = x1_data.to(self.dev)
                x2_data = x2_data.to(self.dev)
                mol_features = mol_features.to(self.dev)
                test_size += x1_data.shape[0]

                x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = self.forward(x1_data, mol_features)
                z2, mu2, logvar2 = self.encode_x2(x2_data)
                x2_rec = self.decode_x2(z2)

                try:
                    x1_array = torch.cat([x1_array, x1_data], dim=0)
                    x1_rec_array = torch.cat([x1_rec_array, x1_rec], dim=0)
                    x2_array = torch.cat([x2_array, x2_data], dim=0)
                    x2_rec_array = torch.cat([x2_rec_array, x2_rec], dim=0)
                    x2_pred_array = torch.cat([x2_pred_array, x2_pred], dim=0)
                    z2_pred_array = torch.cat([z2_pred_array, z2_pred], dim=0)
                    mol_id_array = np.concatenate((mol_id_array, mol_id), axis=0)
                    cp_index_array = np.concatenate((cp_index_array, CP_index), axis=0)

                except:
                    x1_array = x1_data.clone()
                    x1_rec_array = x1_rec.clone()
                    x2_array = x2_data.clone()
                    x2_rec_array = x2_rec.clone()
                    x2_pred_array = x2_pred.clone()
                    z2_pred_array = z2_pred.clone()
                    mol_id_array = mol_id.copy()
                    cp_index_array = CP_index.copy()

        x1_array = x1_array.cpu().numpy().astype(float)
        x1_rec_array = x1_rec_array.cpu().numpy().astype(float)
        x2_array = x2_array.cpu().numpy().astype(float)
        x2_rec_array = x2_rec_array.cpu().numpy().astype(float)
        x2_pred_array = x2_pred_array.cpu().numpy().astype(float)
        z2_pred_array = z2_pred_array.cpu().numpy().astype(float)

        ddict_data = dict()
        ddict_data['x1'] = x1_array
        ddict_data['x2'] = x2_array
        ddict_data['x1_rec'] = x1_rec_array
        ddict_data['x2_rec'] = x2_rec_array
        ddict_data['x2_pred'] = x2_pred_array
        ddict_data['mol_id'] = mol_id_array
        ddict_data['CP_index'] = cp_index_array

        return ddict_data

    def predict_profile_for_x1(self, loader):
        self.eval()
        with torch.no_grad():
            for x1_data, mol_features, mol_id in loader:
                x1_data = x1_data.to(self.dev)
                mol_features = mol_features.to(self.dev)
                x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = self.forward(x1_data, mol_features)
                try:
                    x2_pred_array = torch.cat([x2_pred_array, x2_pred], dim=0)
                    mol_id_array = np.concatenate((mol_id_array, mol_id), axis=0)
                    z2_pred_array = torch.cat([z2_pred_array, z2_pred], dim=0)
                except:
                    x2_pred_array = x2_pred.clone()
                    z2_pred_array = z2_pred.clone()
                    mol_id_array = np.asarray(mol_id).copy()
        x2_pred_array = x2_pred_array.cpu().numpy().astype(float)
        z2_pred_array = z2_pred_array.cpu().numpy().astype(float)
        return x2_pred_array, mol_id_array, z2_pred_array

    def eval_x_reconstruction(self, x1, x1_rec, x2, x2_rec, x2_pred, metrics_func=('pearson',)):
        x1_np = x1.data.cpu().numpy().astype(float)
        x2_np = x2.data.cpu().numpy().astype(float)
        x1_rec_np = x1_rec.data.cpu().numpy().astype(float)
        x2_rec_np = x2_rec.data.cpu().numpy().astype(float)
        x2_pred_np = x2_pred.data.cpu().numpy().astype(float)

        DEG_np = x2_np - x1_np
        DEG_rec_np = x2_rec_np - x1_np
        DEG_pert_np = x2_pred_np - x1_np

        metrics_dict = defaultdict(float)
        metrics_dict_ls = defaultdict(list)
        for m in metrics_func:
            for i in range(x1_np.shape[0]):
                metrics_dict['x1_rec_' + m] += get_metric_func(m)(x1_np[i, :], x1_rec_np[i, :])
                metrics_dict_ls['x1_rec_' + m] += [get_metric_func(m)(x1_np[i, :], x1_rec_np[i, :])]

        for m in metrics_func:
            for i in range(x1_np.shape[0]):
                metrics_dict['x2_rec_' + m] += get_metric_func(m)(x2_np[i, :], x2_rec_np[i, :])
                metrics_dict_ls['x2_rec_' + m] += [get_metric_func(m)(x2_np[i, :], x2_rec_np[i, :])]

        for m in metrics_func:
            for i in range(x1_np.shape[0]):
                metrics_dict['x2_pred_' + m] += get_metric_func(m)(x2_np[i, :], x2_pred_np[i, :])
                metrics_dict_ls['x2_pred_' + m] += [get_metric_func(m)(x2_np[i, :], x2_pred_np[i, :])]

        for m in metrics_func:
            for i in range(x1_np.shape[0]):
                metrics_dict['DEG_rec_' + m] += get_metric_func(m)(DEG_np[i, :], DEG_rec_np[i, :])
                metrics_dict_ls['DEG_rec_' + m] += [get_metric_func(m)(DEG_np[i, :], DEG_rec_np[i, :])]

        for m in metrics_func:
            for i in range(x1_np.shape[0]):
                metrics_dict['DEG_pred_' + m] += get_metric_func(m)(DEG_np[i, :], DEG_pert_np[i, :])
                metrics_dict_ls['DEG_pred_' + m] += [get_metric_func(m)(DEG_np[i, :], DEG_pert_np[i, :])]

        return metrics_dict, metrics_dict_ls

