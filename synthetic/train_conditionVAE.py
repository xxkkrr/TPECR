import sys
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import datetime
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from collections import defaultdict
from LogPrint import Logger

def setup_seed(seed=1521):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_checkpoint(model, trained_epoch):
    save_params = {
        "model": model.state_dict(),
        "trained_epoch": trained_epoch,
    }
    filename = checkpoint_path + f"conditionvae5-epoch{trained_epoch}.pkl"
    torch.save(save_params, filename)

def load_checkpoint(model, trained_epoch):
    filename = checkpoint_path + f"conditionvae5-epoch{trained_epoch}.pkl"
    save_params = torch.load(filename)
    model.load_state_dict(save_params["model"])

class UserDataset(Dataset):
    def __init__(self, user_embed_list, tag_item_list, item_num, tag_num):
        super(UserDataset, self).__init__()
        self.user_embed_list = user_embed_list
        self.tag_item_list = tag_item_list

        self.item_num = item_num
        self.tag_num = tag_num

        self.make_discrete_feature()

    def make_discrete_feature(self):
        print("start make discrete feature")
        self.user_discrete_features = []

        tag_len = torch.tensor([self.item_num // self.tag_num for _ in range(self.tag_num)], dtype=torch.float32)
        tag_matrix = torch.zeros(self.tag_num, self.item_num)
        for _ in range(self.tag_num):
            tag_matrix[_][self.tag_item_list[_]] = 1.
        tag_matrix = tag_matrix.T

        batch = 128
        with torch.no_grad():
            for idx in range(0, len(self.user_embed_list), batch):
                item_score = torch.tensor(self.user_embed_list[idx:idx+batch], dtype=torch.float32)
                mean_score = torch.mean(item_score, -1, keepdim=True)
                tag_score = torch.mm(item_score, tag_matrix)
                tag_score = tag_score / tag_len
                discrete_features = tag_score.to(torch.float32).tolist()
                self.user_discrete_features = self.user_discrete_features + discrete_features

        assert len(self.user_discrete_features) == len(self.user_embed_list)

    def __len__(self):
        return len(self.user_embed_list)

    def __getitem__(self, index):
        return self.user_embed_list[index], self.user_discrete_features[index]

def my_collate_fn(batch):
    batch_input, batch_discrete_feature = zip(*batch)
    batch_input = torch.tensor(batch_input, dtype=torch.float32)
    batch_discrete_feature = torch.tensor(batch_discrete_feature, dtype=torch.float32)
    return batch_input, batch_discrete_feature

def build_loader(cur_dataset, batch_size, num_threads=0, shuffle=True):
    return DataLoader(
        cur_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_threads,
        collate_fn=my_collate_fn
    )

def kl_gaussian(mu,log_var):
    return torch.sum(0.5 * (mu**2 + torch.exp(log_var) - log_var -1))

class condition_VAE5(nn.Module):
    def __init__(self, embed_size, feat_size, h_dim, gaussian_dim, device):
        super(condition_VAE5, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(embed_size+feat_size, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, 2*gaussian_dim)
            )

        self.decoder = nn.Sequential(
            nn.Linear(gaussian_dim+feat_size, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, embed_size))

        self.embed_size = embed_size
        self.feat_size = feat_size
        self.h_dim = h_dim
        self.gaussian_dim = gaussian_dim
        self.device = device

    def to_var(self, x):
        return x.to(self.device)
    
    def reparametrize(self, mu, log_var):
        """"z = mean + eps * sigma where eps is sampled from N(0, 1)."""
        if self.training:
            eps = self.to_var(torch.randn(mu.size(0), mu.size(1)))
        else:
            eps = self.to_var(torch.zeros(mu.size(0), mu.size(1)))
        z = mu + eps * torch.exp(log_var/2)    # 2 for convert var to std
        return z
    
    def forward(self, embed_x, feat_x):
        input_tensor = torch.cat([embed_x, feat_x], dim=-1)
        gauss_latent = self.encoder(input_tensor)
        mues, logs = torch.chunk(gauss_latent, 2, dim=-1)
        mues = mues.contiguous()
        logs = logs.contiguous()
        mus_logs = (mues, logs)
        z = self.reparametrize(mues, logs)
        input_z = torch.cat([z, feat_x], dim=-1)
        out = self.decoder(input_z)
        return out, mus_logs, z

    def generate(self, embed_x, feat_x_enc, feat_x_dec):
        input_tensor = torch.cat([embed_x, feat_x_enc], dim=-1)
        gauss_latent = self.encoder(input_tensor)
        mues, logs = torch.chunk(gauss_latent, 2, dim=-1)
        mues = mues.contiguous()
        logs = logs.contiguous()
        mus_logs = (mues, logs)
        z = self.reparametrize(mues, logs)
        input_z = torch.cat([z, feat_x_dec], dim=-1)
        out = self.decoder(input_z)
        return out, mus_logs, z

    def generate2(self, embed_x, feat_x):
        input_tensor = torch.cat([embed_x, feat_x], dim=-1)
        gauss_latent = self.encoder(input_tensor)
        mues, logs = torch.chunk(gauss_latent, 2, dim=-1)
        mues = mues.contiguous()
        logs = logs.contiguous()
        mus_logs = (mues, logs)
        z = self.reparametrize(mues, logs)
        input_z = torch.cat([z, feat_x], dim=-1)
        out = self.decoder(input_z)
        return out, mus_logs, z


setup_seed()
user_num = 1000
item_num = 50
tag_num = 5
rec_hidden_dim = item_num
use_gpu = True
device = torch.device('cuda') if (torch.cuda.is_available() and use_gpu) else torch.device('cpu')
checkpoint_path = "./conditionvae_checkpoint/"

with open("synthetic2.pkl", "rb") as f: 
    tag_item_list, item_embed_list, user_embed_list = pickle.load(f)
train_test_split_ratio = 0.9
train_test_split_index = int(user_num * train_test_split_ratio)
train_user_dataset = UserDataset(user_embed_list[:train_test_split_index], tag_item_list, item_num, tag_num)
test_user_dataset = UserDataset(user_embed_list[train_test_split_index:], tag_item_list, item_num, tag_num)
train_batch_size = 128
test_batch_size = 128
train_data_loader = build_loader(train_user_dataset, train_batch_size)
test_data_loader = build_loader(test_user_dataset, test_batch_size, shuffle=False)

embed_size = rec_hidden_dim
feat_size = tag_num
h_dim = 16
gaussian_dim = 8
my_vae = condition_VAE5(embed_size, feat_size, h_dim, gaussian_dim, device)
my_vae = my_vae.to(device)

lr = 1e-3
epoch_num = 2000
test_epoch_num = 100
date_str = datetime.date.today().isoformat()
min_beta = 1.
max_beta = 1.
sys.stdout = Logger(f"conditionvae5-pretrain-{date_str}-h-{h_dim}-g-{gaussian_dim}-lr-{lr}-bs-{train_batch_size}-b-linear-min-{min_beta}-max-{max_beta}-tagcore-2.log")

beta = [min_beta] * (epoch_num // 5) + [min_beta+(max_beta-min_beta)/(epoch_num//10)*_ for _ in range(epoch_num//10)]
beta = beta + [max_beta]*(epoch_num - len(beta))

vae_optimizer = torch.optim.Adam(my_vae.parameters(), lr=lr)
mse_loss = nn.MSELoss(reduction='sum')

time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(time_str + " train start")
for _ in range(epoch_num):
    batch_loss_list = []
    batch_rloss_list = []
    batch_klgloss_list = []

    batch_klgloss1_list = []
    batch_klgloss2_list = []

    batch_tklgloss_list = []

    my_vae.train()
    for batch_data in tqdm(train_data_loader, ncols=0):
        batch_input, batch_discrete_feature = batch_data
        batch_input = batch_input.to(device)
        batch_discrete_feature = batch_discrete_feature.to(device)

        out, mus_logs, z = my_vae(batch_input, batch_discrete_feature)
        reconst_loss = mse_loss(out, batch_input)
        kl_gauss_loss =  kl_gaussian(*mus_logs)

        loss = reconst_loss + beta[_] * kl_gauss_loss

        vae_optimizer.zero_grad()
        loss.backward()
        vae_optimizer.step()

        reconst_loss = reconst_loss.detach().cpu().item()
        kl_gauss_loss = kl_gauss_loss.detach().cpu().item()
        batch_loss = loss.detach().cpu().item()
        batch_rloss_list.append(reconst_loss)
        batch_klgloss_list.append(kl_gauss_loss)
        batch_loss_list.append(batch_loss) 

    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    epoch_loss = sum(batch_loss_list) / len(train_user_dataset)
    epoch_rloss = sum(batch_rloss_list) / len(train_user_dataset)
    epoch_klgloss = sum(batch_klgloss_list) / len(train_user_dataset)

    print(f"{time_str} train epoch {_} loss {epoch_loss} rloss {epoch_rloss} klgloss {epoch_klgloss}")

    if (_ + 1) % test_epoch_num == 0:
        my_vae.eval()
        batch_rloss_list = []
        batch_klgloss_list = []
        batch_loss_list = []

        batch_klgloss1_list = []
        batch_klgloss2_list = []

        batch_tklgloss_list = []

        with torch.no_grad():
            for batch_data in tqdm(test_data_loader, ncols=0):
                batch_input, batch_discrete_feature = batch_data
                batch_input = batch_input.to(device)
                batch_discrete_feature = batch_discrete_feature.to(device)

                out, mus_logs, z = my_vae(batch_input, batch_discrete_feature)
                reconst_loss = mse_loss(out, batch_input)
                kl_gauss_loss =  kl_gaussian(*mus_logs)     

                loss = reconst_loss + beta[_] * kl_gauss_loss 

                reconst_loss = reconst_loss.detach().cpu().item()
                kl_gauss_loss = kl_gauss_loss.detach().cpu().item()
                loss = loss.detach().cpu().item()
                batch_rloss_list.append(reconst_loss)
                batch_klgloss_list.append(kl_gauss_loss)
                batch_loss_list.append(loss)

        mean_rloss = sum(batch_rloss_list) / len(test_user_dataset)
        mean_klgloss = sum(batch_klgloss_list) / len(test_user_dataset)
        mean_loss = sum(batch_loss_list) / len(test_user_dataset)

        print(f"{time_str} epoch {_} test loss {mean_loss} rloss {mean_rloss} klgloss {mean_klgloss}")

        save_checkpoint(my_vae, _)