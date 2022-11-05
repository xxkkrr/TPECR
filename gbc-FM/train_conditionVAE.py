import sys
import random
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
from RecModel import ConvRec
from VAE import *

def setup_seed(seed=2252):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def my_collate_fn(batch):
    batch_input, batch_discrete_feature = zip(*batch)
    batch_input = torch.tensor(batch_input)
    batch_discrete_feature = torch.tensor(batch_discrete_feature)
    return batch_input, batch_discrete_feature

class UserDataset(Dataset):
    def __init__(self, user_file, tag_into_file, item_num, tag_num, recmodel, device):
        super(UserDataset, self).__init__()
        self.user_list = []
        with open(user_file, 'r') as f:
            for line in f.readlines():
                user = int(line.strip())
                self.user_list.append(user)
        with open(tag_into_file, 'r') as f:
            ori_tag_info = json.load(f)
        self.tag_info = {int(k):v for k,v in ori_tag_info.items()}

        assert len(self.tag_info) == tag_num
        # assert sum([len(v) for k,v in self.tag_info.items()]) == item_num

        self.item_num = item_num
        self.tag_num = tag_num
        self.recmodel = recmodel
        self.recmodel.eval()
        self.device = device
        self.my_tanh = torch.nn.Tanh()

        self.make_user_embedding()
        self.make_discrete_feature()

    def make_user_embedding(self):
        print("start make user embedding")
        user_idx = torch.tensor(self.user_list).to(self.device)
        self.user_embeddings = self.recmodel.user_embeddings.weight.data[user_idx].cpu().tolist()
        print(self.user_embeddings[0])

    def make_discrete_feature(self):
        print("start make discrete feature")
        self.user_discrete_features = []
        batch = 1024
        with torch.no_grad():
            for idx in range(0, len(self.user_list), batch):
                user_batch = torch.tensor(self.user_list[idx:idx+batch])
                user_batch = user_batch.to(self.device)
                tag_score = self.recmodel.get_all_att_score(user_batch)
                discrete_features = tag_score.to(torch.float32).tolist()
                self.user_discrete_features = self.user_discrete_features + discrete_features

        assert len(self.user_discrete_features) == len(self.user_list)

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, index):
        return self.user_embeddings[index], self.user_discrete_features[index]

def build_loader(cur_dataset, batch_size, num_threads=0, shuffle=True):
    return DataLoader(
        cur_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_threads,
        collate_fn=my_collate_fn
    )


def MAPE(predict, target):
    loss = (predict - target).abs() / (target.abs() + 1e-8)
    return loss.sum()

setup_seed()
user_num = 58318
item_num = 35146
tag_num = 6
rec_hidden_dim = 64
use_gpu = True
device = torch.device('cuda') if (torch.cuda.is_available() and use_gpu) else torch.device('cpu')
checkpoint_path = "./conditionvae_checkpoint/"

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


rec_model = ConvRec(user_num, item_num, tag_num, rec_hidden_dim)
rec_model.to(device)
pretrain_recmodel_path = "./checkpoint/FM-epoch999.pkl"
if use_gpu:
    save_params = torch.load(pretrain_recmodel_path)
    rec_model.load_state_dict(save_params["model"])
else:
    save_params = torch.load(pretrain_recmodel_path, map_location='cpu')
    rec_model.load_state_dict(save_params["model"])
rec_model.eval()


train_user_file = "./data/vae_train_user.txt"
test_user_file = "./data/vae_test_user.txt"
tag_into_file = "./data/tag_info.json"

train_user_dataset = UserDataset(train_user_file, tag_into_file, item_num, tag_num, rec_model, device)
test_user_dataset = UserDataset(test_user_file, tag_into_file, item_num, tag_num, rec_model, device)

train_batch_size = 512
test_batch_size = 512

train_data_loader = build_loader(train_user_dataset, train_batch_size)
test_data_loader = build_loader(test_user_dataset, test_batch_size, shuffle=False)

embed_size = rec_hidden_dim
feat_size = tag_num # tagvae_gaussian_dim
h_dim = 128
gaussian_dim = 48
my_vae = condition_VAE5(embed_size, feat_size, h_dim, gaussian_dim, device)
my_vae = my_vae.to(device)

lr = 2e-4
epoch_num = 4000
test_epoch_num = 50
date_str = datetime.date.today().isoformat()
min_beta = .0001
max_beta = .0001
sys.stdout = Logger(f"conditionvae5-pretrain-{date_str}-h-{h_dim}-g-{gaussian_dim}-lr-{lr}-bs-{train_batch_size}-b-linear-min-{min_beta}-max-{max_beta}-tagcore.log")

beta = [min_beta] * (epoch_num // 20) + [min_beta+(max_beta-min_beta)/(epoch_num//20)*_ for _ in range(epoch_num//20)]
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

    batch_rloss1_list = []
    batch_rloss2_list = []

    my_vae.train()
    for batch_data in tqdm(train_data_loader, ncols=0):
        batch_input, batch_discrete_feature = batch_data
        batch_input = batch_input.to(device)
        batch_discrete_feature = batch_discrete_feature.to(device)

        out, mus_logs, z = my_vae(batch_input, batch_discrete_feature)

        reconst_loss1 = mse_loss(out, batch_input)
        kl_gauss_loss =  kl_gaussian(*mus_logs)
        reconst_loss = reconst_loss1
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

        batch_rloss1_list = []
        batch_rloss2_list = []

        with torch.no_grad():
            for batch_data in tqdm(test_data_loader, ncols=0):
                batch_input, batch_discrete_feature = batch_data
                batch_input = batch_input.to(device)
                batch_discrete_feature = batch_discrete_feature.to(device)

                out, mus_logs, z = my_vae(batch_input, batch_discrete_feature)

                reconst_loss1 = mse_loss(out, batch_input) 
                reconst_loss = reconst_loss1
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