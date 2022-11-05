import sys
import random
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import datetime
import json
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from collections import defaultdict
from LogPrint import Logger
import math
import argparse
import lime
import lime.lime_tabular

def setup_seed(seed=1641):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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

parser = argparse.ArgumentParser(description='test condition vae control')
parser.add_argument('--user', type=str, 
                    help='choose from train or test')
args = parser.parse_args()

setup_seed()
user_num = 1000
item_num = 50
tag_num = 5
rec_hidden_dim = item_num
use_gpu = True
device = torch.device('cuda') if (torch.cuda.is_available() and use_gpu) else torch.device('cpu')

with open("synthetic2.pkl", "rb") as f: 
    tag_item_list, item_embed_list, user_embed_list = pickle.load(f)
train_test_split_ratio = 0.9
train_test_split_index = int(user_num * train_test_split_ratio)
train_user_dataset = UserDataset(user_embed_list, tag_item_list, item_num, tag_num)

embed_size = rec_hidden_dim
feat_size = tag_num
h_dim = 16
gaussian_dim = 8
my_vae = condition_VAE5(embed_size, feat_size, h_dim, gaussian_dim, device)
my_vae = my_vae.to(device)

pretrain_vae_path = "./conditionvae_checkpoint/0408-1632/conditionvae5-epoch499.pkl"
if use_gpu:
    save_params = torch.load(pretrain_vae_path)
    my_vae.load_state_dict(save_params["model"])
else:
    save_params = torch.load(pretrain_vae_path, map_location='cpu')
    my_vae.load_state_dict(save_params["model"])
my_vae.eval()

test_num = 100

if args.user == "train":
    user_idx_list = random.sample([_ for _ in range(train_test_split_index)], test_num)
elif args.user == "test":
    user_idx_list = random.sample([_ for _ in range(train_test_split_index, user_num)], test_num)
else:
    user_idx_list = []
    print("Not support {}. Choose from train or test".format(args.user))


date_str = datetime.date.today().isoformat()
sys.stdout = Logger(f"conditionvae5-limec-{args.user}-result-{date_str}-num-{test_num}.txt")
print(f"pretrain vae path {pretrain_vae_path}")

lime_weight_list = [[] for _ in range(tag_num)]

all_train_data_array = np.array(train_user_dataset.user_discrete_features)
explainer = lime.lime_tabular.LimeTabularExplainer(\
            all_train_data_array, discretize_continuous=False)

class predict_fn:
    def __init__(self, item_id, user_ori_embed, ori_rank, batch_size=64):
        self.item_id = item_id
        self.user_ori_embed = user_ori_embed
        self.ori_rank = ori_rank
        self.batch_size = batch_size

    def __call__(self, input_data_array):
        input_data_tensor = torch.from_numpy(input_data_array.astype('float32'))
        output_data = []
        with torch.no_grad():
            user_ori_embed_tensor = torch.tensor(self.user_ori_embed).to(device)
            for idx in range(0, len(input_data_tensor), self.batch_size):
                batch_discrete_feature = input_data_tensor[idx:idx+self.batch_size].clone().to(device)    
                batch_input = user_ori_embed_tensor.repeat(batch_discrete_feature.size()[0],1)
                out, mus_logs, z = my_vae.generate2(batch_input, batch_discrete_feature) 
                batch_item_score_tensor = out
                batch_item_score_numpy = batch_item_score_tensor.detach().cpu().numpy()
                batch_item_rank_list = np.argsort(batch_item_score_numpy, axis=-1, kind="stable").tolist()

                for item_rank_list in batch_item_rank_list:
                    for rank, item in enumerate(item_rank_list):
                        if item == self.item_id:
                            if rank >= self.ori_rank:
                                output_data.append([0., 1.])
                            else:
                                output_data.append([1., 0.])
                            break       

        return np.array(output_data).astype('float32')


for user_idx in tqdm(user_idx_list):
    for test_tag in range(tag_num):
        test_item = random.choice(tag_item_list[test_tag])  

        user_ori_embed = train_user_dataset.user_embed_list[user_idx]   
        user_discrete_feature = train_user_dataset.user_discrete_features[user_idx]
        user_ori_embed_tensor = torch.tensor(user_ori_embed).to(device)
        user_ori_user_discrete_tensor = torch.tensor(user_discrete_feature).to(device) 

        batch_input = user_ori_embed_tensor.unsqueeze(0)
        batch_ori_discrete_feature = user_ori_user_discrete_tensor.unsqueeze(0) 
        with torch.no_grad():
            ori_out, mus_logs, z = my_vae.generate2(batch_input, batch_ori_discrete_feature)   
            ori_out = ori_out.squeeze(0)
            ori_item_score_tensor = ori_out
        ori_item_score_array = ori_item_score_tensor.detach().cpu().numpy()
        rank_list = np.argsort(ori_item_score_array, kind="stable").tolist()
        ori_rank = None
        for rank, item_id in enumerate(rank_list):
            if item_id == test_item:
                ori_rank = rank
                break

        exp = explainer.explain_instance(np.array(user_discrete_feature), predict_fn(test_item, user_ori_embed, ori_rank), num_features=tag_num)
        weight_list = [0. for _ in range(tag_num)]
        for fea_name, value in exp.as_list():
            weight_list[int(fea_name)] = value
        lime_weight_list[test_tag].append(weight_list)

lime_result_list = [np.array(weights).mean(axis=0) for weights in lime_weight_list]
print(f"lime_result_list: {lime_result_list}")