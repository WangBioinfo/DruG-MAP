"""
@File   :   model_x1
@Desc   :   Reconstruction model for Control profile data
"""
import torch.nn.functional as F
from utils import *
from collections import defaultdict
from torch import nn, optim
from torch.autograd import grad

class Model_x1(torch.nn.Module):
    def __init__(self, n_features, n_latent, n_en_hidden, n_de_hidden, BCE_pos_weight, **kwargs):
        super(Model_x1, self).__init__()
        self.n_features = n_features
        self.n_latent = n_latent
        self.n_en_hidden = n_en_hidden
        self.n_de_hidden = n_de_hidden
        self.init_w = kwargs.get('init_w', False)
        self.beta = kwargs.get('beta', 0.05)
        self.path_model = kwargs.get('path_model', "pretrain_model")
        self.dev = kwargs.get('device', torch.device('cpu'))
        self.dropout = kwargs.get('dropout', 0.3)
        self.BCE_pos_weight = torch.Tensor([BCE_pos_weight]).to(self.dev) if BCE_pos_weight != None else None
        self.random_seed = kwargs.get('random_seed', 20240729)
        self.n_critic = kwargs.get('n_critic')
        encoder = [
            nn.Linear(self.n_features, self.n_en_hidden[0]),
            nn.BatchNorm1d(self.n_en_hidden[0]),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        ]

        if len(n_en_hidden) > 1:
            for i in range(len(n_en_hidden) - 1):
                en_hidden = [
                    nn.Linear(self.n_en_hidden[i], self.n_en_hidden[i + 1]),
                    nn.BatchNorm1d(self.n_en_hidden[i + 1]),
                    nn.LeakyReLU(),
                    nn.Dropout(self.dropout)
                ]
                encoder = encoder + en_hidden
        self.encoder_x1 = nn.Sequential(*encoder)
        self.mu_z1 = nn.Sequential(nn.Linear(self.n_en_hidden[-1], self.n_latent), )
        self.logvar_z1 = nn.Sequential(nn.Linear(self.n_en_hidden[-1], self.n_latent), )

        if len(n_de_hidden) == 0:
            decoder = [nn.Linear(self.n_latent, self.n_features)]
        else:
            decoder = [
                nn.Linear(self.n_latent, self.n_de_hidden[0]),
                nn.BatchNorm1d(self.n_de_hidden[0]),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout)
            ]

            if len(n_de_hidden) > 1:
                for i in range(len(self.n_de_hidden) - 1):
                    de_hidden = [
                        nn.Linear(self.n_de_hidden[i], self.n_de_hidden[i + 1]),
                        nn.BatchNorm1d(self.n_de_hidden[i + 1]),
                        nn.LeakyReLU(),
                        nn.Dropout(self.dropout)
                    ]
                    decoder = decoder + de_hidden
            decoder.append(nn.Linear(self.n_de_hidden[-1], self.n_features))

        self.decoder_x1 = nn.Sequential(*decoder)

        if self.init_w:
            self.encoder_x1.apply(self._init_weights)
            self.decoder_x1.apply(self._init_weights)

    def _init_weights(self, layer):
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
        return

    def encode(self, x1):
        y = self.encoder_x1(x1)
        mu, logvar = self.mu_z1(y), self.logvar_z1(y)
        z = self.sample_latent(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        x1_rec = self.decoder_x1(z)
        return x1_rec

    def sample_latent(self, mu, logvar):
        # Reparametrization
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.dev)
        eps = eps.mul_(std).add_(mu)
        return eps

    def _average_latent(self, x1):
        z, _, _ = self.encode(x1)
        mean_z = z.mean(0)
        return mean_z

    def forward(self, x1):
        # Forward pass through full network
        z, mu, logvar = self.encode(x1)
        rec_x1 = self.decode(z)
        return rec_x1, mu, logvar

    def loss(self, y_pred, y_true, mu, logvar):
        kld = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp(), )
        mse = F.mse_loss(y_pred, y_true, reduction="sum")
        return mse + self.beta * kld, mse, kld

    def train_model(self, learning_rate, weight_decay, n_epochs, train_loader, test_loader, save_model=True,
                    metrics_func=None):
        D = Discriminator(n_features=self.n_features, dropout=self.dropout, dev=self.dev)
        D.to(self.dev)
        epoch_hist = defaultdict(list)
        optimizer_enc_dec = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
        optimizer_disc = optim.Adam(D.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
        loss_item = ['loss', 'mse_x1', 'kld_x1', 'loss_D', 'loss_G']
        # Train model
        best_value = np.inf
        best_epoch = 0
        for epoch in range(n_epochs):
            train_size = 0
            loss_value = 0
            self.train()
            for i, (x1_train, CP_index, mol_id) in enumerate(train_loader):
                x1_train = x1_train.to(self.dev)
                if x1_train.shape[0] == 1:
                    continue
                train_size += x1_train.shape[0]
                # Pass through full network
                x1_rec, x1_mu, x1_logvar = self.forward(x1_train)
                optimizer_disc.zero_grad()
                # Train discriminator
                outputs_real = D.forward(x1_train)
                outputs_fake = D.forward(x1_rec.detach())
                # Total loss of discriminator
                lambda_gp = 10
                loss_disc = D.dis_loss(x1_train, x1_rec.detach(), outputs_real, outputs_fake, lambda_gp)
                # Backward
                loss_disc.backward()
                optimizer_disc.step()

                # Train the encoder and decoder (as generators)
                outputs_fake_gen = D.forward(x1_rec)
                # The generator's goal is to maximize the score of D (i.e., minimize the score of -D)
                loss_gen = D.gen_loss(outputs_fake_gen)

                # Calculate the reconstruction loss value of VAE
                optimizer_enc_dec.zero_grad()
                loss, _, kld = self.loss(x1_rec, x1_train, x1_mu, x1_logvar)
                # Calculate the total loss value
                loss_vae_gan = loss_gen + 1e-3 * loss
                loss_value += loss.item()
                loss_vae_gan.backward()
                optimizer_enc_dec.step()

            # Test using the train dataset
            train_dict, train_metrics_dict, train_metrics_dict_ls = self.test_model(loader=train_loader,
                                                                                    loss_item=loss_item,
                                                                                    metrics_func=metrics_func)
            # Extract loss data
            train_loss = train_dict['loss']
            train_mse_x1 = train_dict['mse_x1']
            train_kld_x1 = train_dict['kld_x1']
            train_loss_D = train_dict['loss_D']
            train_loss_G = train_dict['loss_G']
            # Integrate training loss data
            for k, v in train_dict.items():
                epoch_hist['train_' + k].append(v)

            for k, v in train_metrics_dict.items():
                epoch_hist['train_' + k].append(v)

            # Test using the valid dataset
            test_dict, test_metrics_dict, test_metricsdict_ls = self.test_model(loader=test_loader, loss_item=loss_item,
                                                                                metrics_func=metrics_func)
            test_loss = test_dict['loss']
            test_mse_x1 = test_dict['mse_x1']
            test_kld_x1 = test_dict['kld_x1']
            test_loss_D = train_dict['loss_D']
            test_loss_G = train_dict['loss_G']
            for k, v in test_dict.items():
                epoch_hist['valid_' + k].append(v)
            for k, v in test_metrics_dict.items():
                epoch_hist['valid_' + k].append(v)

            # Print the loss value
            print(f'[Epoch {epoch}] | loss: {train_loss:.3f}, mse_x1_rec: {train_mse_x1:.3f}, kld_x1: {train_kld_x1:.3f}, train_loss_D: {train_loss_D:.5f}, train_loss_G: {train_loss_G:.5f}| valid_loss: {test_loss:.3f}, valid_mse_x1_rec: {test_mse_x1:.3f}, valid_kld_x1: {test_kld_x1:.3f}, test_loss_D: {test_loss_D:.5f}, test_loss_G: {test_loss_G:.5f} |')

            # If the validation loss value is smaller, means the current model performs better, should be saved
            if test_loss < best_value:
                best_value = test_loss
                best_epoch = epoch
                if save_model:
                    torch.save(self, self.path_model + 'best_model.pt')

        return epoch_hist, best_epoch

    def test_model(self, loader, loss_item=None, metrics_func=None):
        D = Discriminator(n_features=self.n_features, dropout=self.dropout, dev=self.dev)
        D.to(self.dev)
        test_dict_loss = defaultdict(float)
        metrics_dict_all = defaultdict(float)
        metrics_dict_all_ls = defaultdict(list)
        test_size = 0
        x1_array = []
        x1_rec_array = []
        mol_id_array = []
        cp_index_array = []
        self.eval()
        with torch.no_grad():
            for x1_train, CP_index, mol_id in loader:
                x1_data = x1_train.to(self.dev)
                CP_index = np.array(list(CP_index))
                mol_id = np.array(list(mol_id))
                test_size += x1_data.shape[0]

                x1_rec, x1_mu, x1_logvar = self.forward(x1_data)
                loss_ls = self.loss(x1_rec, x1_data, x1_mu, x1_logvar)

                outputs_real = D.forward(x1_data)
                outputs_fake = D.forward(x1_rec.detach())
                loss_D = -D.wasserstein_loss(outputs_real) + D.wasserstein_loss(outputs_fake)
                outputs_fake_gen = D.forward(x1_rec)
                loss_G = D.gen_loss(outputs_fake_gen)

                loss_ls = list(loss_ls)
                loss_ls.append(loss_D)
                loss_ls.append(loss_G)

                # Integrate loss data into test_dict
                if loss_item != None:
                    for idx, k in enumerate(loss_item):
                        test_dict_loss[k] += loss_ls[idx].item()

                if metrics_func != None:
                    metrics_dict, metrics_dict_ls = self.eval_x1_reconstruction(x1_data, x1_rec,
                                                                                metrics_func=metrics_func)
                    for k in metrics_dict.keys():
                        metrics_dict_all[k] += metrics_dict[k]
                    for k in metrics_dict_ls.keys():
                        metrics_dict_all_ls[k] += metrics_dict_ls[k]
                    # Integrate metadata
                    metrics_dict_all_ls['mol_id'] += list(mol_id)
                    metrics_dict_all_ls['CP_index'] += list(CP_index)

                try:
                    x1_array = torch.cat([x1_array, x1_data], dim=0)
                    x1_rec_array = torch.cat([x1_rec_array, x1_rec], dim=0)
                    mol_id_array = np.concatenate([mol_id_array, mol_id], axis=0)
                    cp_index_array = np.concatenate((cp_index_array, CP_index), axis=0)
                except:
                    x1_array = x1_data.clone()
                    x1_rec_array = x1_rec.clone()
                    mol_id_array = mol_id.copy()
                    cp_index_array = CP_index.copy()
        # Get mean loss
        for k in test_dict_loss.keys():
            test_dict_loss[k] = test_dict_loss[k] / test_size
        for k in metrics_dict_all.keys():
            metrics_dict_all[k] = metrics_dict_all[k] / test_size
        return test_dict_loss, metrics_dict_all, metrics_dict_all_ls

    def eval_x1_reconstruction(self, x1, x1_rec, metrics_func=['r2']):
        # Initial value
        x1_np = x1.data.cpu().numpy().astype(float)
        # Reconstructed value
        x1_rec_np = x1_rec.data.cpu().numpy().astype(float)

        metrics_dict = defaultdict(float)
        metrics_dict_ls = defaultdict(list)
        for m in metrics_func:
            for i in range(x1_np.shape[0]):
                metrics_dict['x1_rec_' + m] += get_metric_func(m)(x1_np[i, :], x1_rec_np[i, :])
                metrics_dict_ls['x1_rec_' + m] += [get_metric_func(m)(x1_np[i, :], x1_rec_np[i, :])]

        return metrics_dict, metrics_dict_ls

    def recon_profile(self, loader):
        test_size = 0
        self.eval()
        with torch.no_grad():
            for x1_train, CP_index, mol_id in loader:
                x1_data = x1_train.to(self.dev)
                CP_index = np.array(list(CP_index))
                mol_id = np.array(list(mol_id))
                test_size += x1_data.shape[0]

                x1_rec, mu1, logvar1 = self.forward(x1_data)

                try:
                    x1_array = torch.cat([x1_array, x1_data], dim=0)
                    x1_rec_array = torch.cat([x1_rec_array, x1_rec], dim=0)
                    mol_id_array = np.concatenate((mol_id_array, mol_id), axis=0)
                    cp_index_array = np.concatenate((cp_index_array, CP_index), axis=0)

                except:
                    x1_array = x1_data.clone()
                    x1_rec_array = x1_rec.clone()
                    mol_id_array = mol_id.copy()
                    cp_index_array = CP_index.copy()

        x1_array = x1_array.cpu().numpy().astype(float)
        x1_rec_array = x1_rec_array.cpu().numpy().astype(float)

        ddict_data = dict()
        ddict_data['x1'] = x1_array
        ddict_data['x1_rec'] = x1_rec_array
        ddict_data['mol_id'] = mol_id_array
        ddict_data['CP_index'] = cp_index_array

        for k in ddict_data.keys():
            print(f'{type(ddict_data[k][0])}, {ddict_data[k].shape}')
        return ddict_data

class Discriminator(nn.Module):
    def __init__(self, n_features, dropout, dev):
        self.n_features = n_features
        self.dropout = dropout
        self.dev = dev
        super(Discriminator, self).__init__()
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(self.n_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )

    # Forward of discriminator
    def forward(self, z):
        z.to(self.dev)
        discrim = self.discriminator(z)
        return discrim

    # Loss function
    def gen_loss(self, outputs_fake_gen):
        return -self.wasserstein_loss(outputs_fake_gen)

    def dis_loss(self, raw, rec, outputs_real, outputs_fake, lambda_gp):
        gradient_penalty = self.compute_gradient_penalty(raw, rec)
        return -self.wasserstein_loss(outputs_real) + self.wasserstein_loss(
            outputs_fake) + lambda_gp * gradient_penalty

    def wasserstein_loss(self, y_pred):
        return torch.mean(y_pred)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1).expand_as(real_samples).to(self.dev)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.forward(interpolates)
        gradients = grad(outputs=d_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(d_interpolates.size()).to(self.dev),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty