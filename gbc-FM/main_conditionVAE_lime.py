import sys
import random
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import datetime
import json
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from collections import defaultdict
from LogPrint import Logger
from utils import *
from RecModel import *
from VAE import *
import math
import argparse
import lime
import lime.lime_tabular

def setup_seed(seed=2049):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def cal_mean_and_var(l):
    l_array = np.array(l)
    return l_array.mean(), l_array.std()

def cal_norm(l, norm=1):
    if norm == 1:
        return sum([abs(_) for _ in l])
    if norm == 2:
        return math.sqrt(sum([_*_ for _ in l]))

def min_max_norm(l):
    min_l = min(l)
    max_l = max(l)
    return [(_ - min_l) / (max_l - min_l) for _ in l]

def zero_mean_norm(l):
    m, v = cal_mean_and_var(l)
    return [(_-m)/v for _ in l]

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

        self.name2id = {}
        for idx, name in enumerate(self.user_list):
            self.name2id[name] = idx

    def make_user_embedding(self):
        print("start make user embedding")
        user_idx = torch.tensor(self.user_list).to(self.device)
        self.user_embeddings = self.recmodel.user_embeddings.weight.data[user_idx].cpu().tolist()

    def make_discrete_feature(self):
        print("start make discrete feature")
        self.user_discrete_features = []
        # self.user_item_score = []
        self.user_01_features = []
        self.user_ori_score = []

        batch = 128
        with torch.no_grad():
            for idx in range(0, len(self.user_list), batch):
                user_batch = torch.tensor(self.user_list[idx:idx+batch])
                user_batch = user_batch.to(self.device)

                tag_score = self.recmodel.get_all_att_score(user_batch)
                mean_score = torch.mean(tag_score, -1, keepdim=True)
                discrete_features = tag_score.to(torch.float32).tolist()
                
                discrete_features = (tag_score > mean_score).to(torch.int).tolist()
                self.user_01_features = self.user_01_features + discrete_features
                self.user_ori_score = self.user_ori_score + tag_score.to(torch.float32).tolist()
                discrete_features = tag_score.to(torch.float32).tolist()
                self.user_discrete_features = self.user_discrete_features + discrete_features

        assert len(self.user_discrete_features) == len(self.user_list)
        # assert len(self.user_item_score) == len(self.user_list)
        assert len(self.user_01_features) == len(self.user_list)
        assert len(self.user_ori_score) == len(self.user_list)

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, index):
        return self.user_embeddings[index], self.user_discrete_features[index]+[1]

class static_list():
    def __init__(self):
        self.mean_list = []
        self.var_list = []

    def add(self, mean, var):
        self.mean_list.append(mean)
        self.var_list.append(var)

    def get_result(self):
        all_mean = sum(self.mean_list) / len(self.mean_list)
        all_var = sum(self.var_list) / len(self.var_list)
        return all_mean, all_var

def update_static_list(each_critique_ap_diff, each_critique_ndcg_diff, each_critique_recall_diff, rank_diff_list, ap, ndcg, recall, rank):
    for idx, each_ap in enumerate(each_critique_ap_diff):
        mean, var = cal_mean_and_var(each_ap)
        ap[idx].append(mean)

    for idx, each_ndcg in enumerate(each_critique_ndcg_diff):
        mean, var = cal_mean_and_var(each_ndcg)
        ndcg[idx].append(mean)

    for idx, each_recall in enumerate(each_critique_recall_diff):
        mean, var = cal_mean_and_var(each_recall)
        recall[idx].append(mean)

    for idx, each_rank in enumerate(rank_diff_list):
        rank[idx].append(each_rank)  

def print_static_list(ap, ndcg, recall, rank, cur_name):
    ap_list = [cal_mean_and_var(_) for _ in ap]
    ndcg_list = [cal_mean_and_var(_) for _ in ndcg]
    recall_list = [cal_mean_and_var(_) for _ in recall]
    rank_list = [cal_mean_and_var(_) for _ in rank]
    print(f"----------\n{cur_name} result:\nap:{ap_list}\nndcg:{ndcg_list}\nrercall:{recall_list}\nrank:{rank_list}")

def load_user_item_pair(file_path):
    user_item_pair = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            if len(line) == 0:
                break
            user, item = line.strip('\n').split(',')
            user = int(user)
            item = int(item)
            user_item_pair.append([user, item])
    return user_item_pair

setup_seed()

with open("./data/user_info.json", "r") as f:
    user_info = json.load(f)
# user_info = {int(k): v for k, v in user_info.items()}
with open("./data/item_info.json", "r") as f:
    item_info = json.load(f)
item_info = {int(k): v for k, v in item_info.items()}

user_num = 58318
item_num = 35146
tag_num = 6
rec_hidden_dim = 64
use_gpu = True
device = torch.device('cuda') if (torch.cuda.is_available() and use_gpu) else torch.device('cpu')

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

test_num = 10000
user_mode = "train"

train_user_file = "./data/vae_train_user.txt"
test_user_file = "./data/vae_test_user.txt"
tag_into_file = "./data/tag_info.json"
train_user_dataset = UserDataset(train_user_file, tag_into_file, item_num, tag_num, rec_model, device)
# test_user_dataset = UserDataset(test_user_file, tag_into_file, item_num, tag_num, rec_model, device)

if user_mode == "train":
    test_file = "./data/vae_test_user_item.txt"
else:
    test_file = "./data/vae_test_user_item_unseen.txt"
test_user_item_pair = load_user_item_pair(test_file)
random.shuffle(test_user_item_pair)


embed_size = rec_hidden_dim
feat_size = tag_num # tagvae_gaussian_dim 
# feat_size = tag_num
# h_dim = 1024
# # gaussian_dim = 128 #256
# gaussian_dim = 128
h_dim = 128
gaussian_dim = 48
my_vae = condition_VAE5(embed_size, feat_size, h_dim, gaussian_dim, device)
my_vae = my_vae.to(device)
pretrain_vae_path = "./conditionvae_checkpoint/conditionvae5-epoch1499.pkl"
if use_gpu:
    save_params = torch.load(pretrain_vae_path)
    my_vae.load_state_dict(save_params["model"])
else:
    save_params = torch.load(pretrain_vae_path, map_location='cpu')
    my_vae.load_state_dict(save_params["model"])
my_vae.eval()
for para in my_vae.parameters():
    para.requires_grad = False

neg_num = 99
user_info = {int(k): [_ for _ in v] for k, v in user_info.items()}


rec_model.fix_item_para()


date_str = datetime.date.today().isoformat()
sys.stdout = Logger(f"conditionvae5-limec-{date_str}-num-{test_num}-attscore-onetag-user-{user_mode}-neg-{neg_num}.txt")
print(f"pretrain vae path {pretrain_vae_path} user_mode {user_mode}")

all_train_data_array = np.array(train_user_dataset.user_discrete_features)
explainer = lime.lime_tabular.LimeTabularExplainer(\
            all_train_data_array, discretize_continuous=False)

class predict_fn:
    def __init__(self, item_id, user_ori_embed, ori_rank, all_item_list, batch_size=64):
        self.item_id = item_id
        self.user_ori_embed = user_ori_embed
        self.ori_rank = ori_rank
        self.all_item_list = all_item_list
        self.batch_size = batch_size

    def __call__(self, input_data_array):
        input_data_tensor = torch.from_numpy(input_data_array.astype('float32'))
        output_data = []
        with torch.no_grad():
            user_ori_embed_tensor = torch.tensor(self.user_ori_embed).to(device)
            all_item_list_tensor = torch.tensor(self.all_item_list).to(device)
            for idx in range(0, len(input_data_tensor), self.batch_size):
                batch_discrete_feature = input_data_tensor[idx:idx+self.batch_size].clone().to(device)    
                # batch_input = user_ori_embed_tensor.unsqueeze(0).expand_as(batch_discrete_feature)
                batch_input = user_ori_embed_tensor.repeat(batch_discrete_feature.size()[0],1)
                out, mus_logs, z = my_vae.generate2(batch_input, batch_discrete_feature) 
                batch_item_score_tensor = rec_model.get_batch_item_score(out, all_item_list_tensor)
                batch_item_rank_list = batch_item_score_tensor.argsort(dim=-1, descending=True).detach().cpu().tolist()
                for item_rank_list in batch_item_rank_list:
                    for rank, item in enumerate(item_rank_list):
                        if item == 0:
                            # print(f"rank: {rank}")
                            if rank <= self.ori_rank:
                                output_data.append([0., 1.])
                                # print("if")
                            else:
                                # print("else")
                                output_data.append([1., 0.])
                            break                 
               
        return np.array(output_data).astype('float32')



fenduan = [(-1e10,-.3)] + [(_/100, (_+3)/100) for _ in range(-30,30,3)] + [(.3,1e10)]
print(f"fenduan {fenduan}")
item_tag_fenbu = [0 for _ in range(len(fenduan))]
other_tag_fenbu = [0 for _ in range(len(fenduan))]

for user, item in tqdm(test_user_item_pair[:test_num]): 

    user_idx = train_user_dataset.name2id[user]
    user_ori_embed = train_user_dataset.user_embeddings[user_idx]   

    user_discrete_feature = train_user_dataset.user_discrete_features[user_idx] 

    ori_user_discrete_feature = user_discrete_feature.copy()    

    user_01_feature = train_user_dataset.user_01_features[user_idx] 

    neg_items = set(range(item_num)) - set([item])
    assert item not in neg_items
    neg_items_sample = random.sample(list(neg_items), neg_num)
    all_item_list = [item] + neg_items_sample

    user_ori_embed_tensor = torch.tensor(user_ori_embed).to(device)
    user_ori_user_discrete_tensor = torch.tensor(ori_user_discrete_feature).to(device) 
    all_item_list_tensor = torch.tensor(all_item_list).to(device)
    # user_ori_user_discrete_tensor.requires_grad = True

    batch_input = user_ori_embed_tensor.unsqueeze(0)
    batch_ori_discrete_feature = user_ori_user_discrete_tensor.unsqueeze(0) 
    with torch.no_grad():
        ori_out, mus_logs, z = my_vae.generate2(batch_input, batch_ori_discrete_feature)   
        ori_out = ori_out.squeeze(0)
        ori_item_score_tensor = rec_model.get_item_score(ori_out, all_item_list_tensor)  
    rank_list = ori_item_score_tensor.argsort(dim=-1, descending=True).detach().cpu().tolist()
    ori_rank = None
    for rank, item_id in enumerate(rank_list):
        if item_id == 0:
            ori_rank = rank
            break

    exp = explainer.explain_instance(np.array(user_discrete_feature), predict_fn(item, user_ori_embed, ori_rank, all_item_list), num_features=tag_num)
    weight_list = [0. for _ in range(tag_num)]
    for fea_name, value in exp.as_list():
        weight_list[int(fea_name)] = value
    ori_grad_list = weight_list

    for tag_idx, grad_val in enumerate(ori_grad_list):
        inter_val_id = None
        for idx, (zuo, you) in enumerate(fenduan):
            if grad_val >= zuo and grad_val <= you:
                inter_val_id = idx
                break
        if tag_idx in item_info[item]:
            item_tag_fenbu[inter_val_id] += 1
        else:
            other_tag_fenbu[inter_val_id] += 1

print(f"item_tag_fenbu {item_tag_fenbu}")
print(f"other_tag_fenbu {other_tag_fenbu}")