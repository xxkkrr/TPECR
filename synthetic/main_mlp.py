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

def setup_seed(seed=1641):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_checkpoint(model, trained_epoch):
    filename = checkpoint_path + f"mlp-epoch{trained_epoch}.pkl"
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


class MLP(nn.Module):
    def __init__(self, embed_size, feat_size, h_dim, device):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(embed_size+feat_size, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, embed_size)
            )

        self.embed_size = embed_size
        self.feat_size = feat_size
        self.h_dim = h_dim
        self.device = device
    
    def forward(self, embed_x, feat_x):
        input_tensor = torch.cat([embed_x, feat_x], dim=-1)
        return self.layers(input_tensor)

    def generate2(self, embed_x, feat_x):
        return self.forward(embed_x, feat_x)

def calculate_ndcg_score(item_score_numpy, gt_item_score_numpy, topk=None):
    ndcg = metrics.ndcg_score(gt_item_score_numpy.reshape(1, -1), item_score_numpy.reshape(1, -1), k=topk)
    return ndcg

def calculate_mse(item_score_numpy, gt_item_score_numpy):
    return np.square(item_score_numpy - gt_item_score_numpy).mean()

def calculate_mape(item_score_numpy, gt_item_score_numpy):
    return np.mean(np.abs((gt_item_score_numpy - item_score_numpy) / gt_item_score_numpy)) * 100

# https://github.com/scikit-learn/scikit-learn/issues/17639#issuecomment-647459706
def calculate_ndcg_score_with_neg_score(item_score_numpy, gt_item_score_numpy, topk=None):
    y_true = np.array(sorted(gt_item_score_numpy,reverse=True)).reshape(1, -1)
    y_true2 = np.array(sorted(gt_item_score_numpy,reverse=False)).reshape(1, -1)
    max_dcg = metrics.dcg_score(y_true, y_true)
    min_dcg =  metrics.dcg_score(y_true, y_true2)

    if max_dcg < min_dcg:
        print("max_dcg < min_dcg")
        print(gt_item_score_numpy)

    y_true = gt_item_score_numpy.reshape(1, -1)
    y_score = item_score_numpy.reshape(1, -1)
    actual_dcg = metrics.dcg_score(y_true, y_score)

    if actual_dcg < min_dcg:
        print("actual_dcg < min_dcg")
        print(item_score_numpy, gt_item_score_numpy)
        print(y_score, y_true)

    return (actual_dcg - min_dcg) / (max_dcg - min_dcg)

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
checkpoint_path = "./mlp_checkpoint/"

with open("synthetic2.pkl", "rb") as f: 
    tag_item_list, item_embed_list, user_embed_list = pickle.load(f)
train_test_split_ratio = 0.9
train_test_split_index = int(user_num * train_test_split_ratio)
train_user_dataset = UserDataset(user_embed_list, tag_item_list, item_num, tag_num)

embed_size = rec_hidden_dim
feat_size = tag_num
h_dim = 16  
my_vae = MLP(embed_size, feat_size, h_dim, device)
my_vae = my_vae.to(device)

pretrain_vae_path = "./mlp_checkpoint/mlp-epoch1599.pkl"
if use_gpu:
    save_params = torch.load(pretrain_vae_path)
    my_vae.load_state_dict(save_params["model"])
else:
    save_params = torch.load(pretrain_vae_path, map_location='cpu')
    my_vae.load_state_dict(save_params["model"])
my_vae.eval()


ndcg_ori_list = [[] for _ in range(tag_num)]
ndcg_result_list = [[] for _ in range(tag_num)]
mse_ori_list = [[] for _ in range(tag_num)]
mse_result_list = [[] for _ in range(tag_num)]
mape_ori_list = [[] for _ in range(tag_num)]
mape_result_list = [[] for _ in range(tag_num)]
test_num = 100

if args.user == "train":
    user_idx_list = random.sample([_ for _ in range(train_test_split_index)], test_num)
elif args.user == "test":
    user_idx_list = random.sample([_ for _ in range(train_test_split_index, user_num)], test_num)
else:
    user_idx_list = []
    print("Not support {}. Choose from train or test".format(args.user))


date_str = datetime.date.today().isoformat()
sys.stdout = Logger(f"mlp-{args.user}-result-{date_str}-num-{test_num}-2.txt")
print(f"pretrain vae path {pretrain_vae_path}")

for user_idx in user_idx_list:
    user_ori_embed = train_user_dataset.user_embed_list[user_idx]   
    user_discrete_feature = train_user_dataset.user_discrete_features[user_idx]

    user_ori_embed_tensor = torch.tensor(user_ori_embed, dtype=torch.float32).to(device)
    user_ori_user_discrete_tensor = torch.tensor(user_discrete_feature, dtype=torch.float32).to(device)  
    batch_input = user_ori_embed_tensor.unsqueeze(0)
    batch_ori_discrete_feature = user_ori_user_discrete_tensor.unsqueeze(0) 

    for num in range(1, 2**tag_num):
        binary = str(bin(num))[2:]
        if len(binary) < tag_num:
            binary = '0' * (tag_num - len(binary)) + binary
        one_count = 0

        gt_item_score = user_ori_embed.copy()
        cur_user_discrete_feature = user_discrete_feature.copy()

        for tag_idx in range(tag_num):
            if binary[tag_idx] == '0':
                continue
            one_count += 1
            new_tag_score = random.random()
            cur_user_discrete_feature[tag_idx] = new_tag_score
            for i in tag_item_list[tag_idx]:
                gt_item_score[i] = new_tag_score
        gt_item_score = np.array(gt_item_score)

        user_discrete_feature_tensor = torch.tensor(cur_user_discrete_feature, dtype=torch.float32).to(device)
        batch_discrete_feature = user_discrete_feature_tensor.unsqueeze(0)
        with torch.no_grad():
            out = my_vae.generate2(batch_input, batch_discrete_feature)
        out = out.squeeze(0).detach().cpu().numpy()        

        ndcg = calculate_ndcg_score(out, gt_item_score)
        ndcg_result_list[one_count-1].append(ndcg)
        mse = calculate_mse(out, gt_item_score)
        mse_result_list[one_count-1].append(mse)
        mape = calculate_mape(out, gt_item_score)
        mape_result_list[one_count-1].append(mape)

        ori_ndcg = calculate_ndcg_score(np.array(user_ori_embed, dtype=np.float32), gt_item_score)
        ndcg_ori_list[one_count-1].append(ori_ndcg)
        ori_mse = calculate_mse(np.array(user_ori_embed, dtype=np.float32), gt_item_score)
        mse_ori_list[one_count-1].append(ori_mse)
        ori_mape = calculate_mape(np.array(user_ori_embed, dtype=np.float32), gt_item_score)
        mape_ori_list[one_count-1].append(ori_mape)

print(f"ori ndcg {[sum(_)/len(_) for _ in ndcg_ori_list]} mse {[sum(_)/len(_) for _ in mse_ori_list]} mape {[sum(_)/len(_) for _ in mape_ori_list]}")
print(f"result ndcg {[sum(_)/len(_) for _ in ndcg_result_list]} mse {[sum(_)/len(_) for _ in mse_result_list]} mape {[sum(_)/len(_) for _ in mape_result_list]} ")