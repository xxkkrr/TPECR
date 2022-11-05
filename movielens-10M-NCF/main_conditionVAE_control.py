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
from RecModel import *
from VAE import *
from utils import *
import math
import argparse

def setup_seed(seed=1647):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def cal_mean_and_var(l):
    l_array = np.array(l);
    return l_array.mean(), l_array.std();

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
        self.user_01_features = []
        self.user_ori_score = []

        tag_len = torch.tensor([len(self.tag_info[_]) for _ in range(self.tag_num)])
        tag_matrix = torch.zeros(self.tag_num, self.item_num)
        for _ in range(self.tag_num):
            tag_matrix[_][self.tag_info[_]] = 1.
        tag_matrix = tag_matrix.T

        self.tag_matrix = tag_matrix
        self.tag_len = tag_len

        batch = 128
        with torch.no_grad():
            for idx in range(0, len(self.user_list), batch):
                user_batch = torch.tensor(self.user_list[idx:idx+batch])
                user_batch = user_batch.to(self.device)
                item_score = self.recmodel.get_batch_item_score2(user_batch)
                item_score = item_score.cpu()
                
                mean_score = torch.mean(item_score, -1, keepdim=True)
                tag_score = torch.mm(item_score, tag_matrix)
                tag_score = tag_score / tag_len
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
        ap[idx].add(mean, var)

    for idx, each_ndcg in enumerate(each_critique_ndcg_diff):
        mean, var = cal_mean_and_var(each_ndcg)
        ndcg[idx].add(mean, var)

    for idx, each_recall in enumerate(each_critique_recall_diff):
        mean, var = cal_mean_and_var(each_recall)
        recall[idx].add(mean, var)

    for idx, each_rank in enumerate(rank_diff_list):
        rank[idx].append(each_rank)  


def print_static_list(ap, ndcg, recall, rank, cur_name):
    ap_list = [_.get_result() for _ in ap]
    ndcg_list = [_.get_result() for _ in ndcg]
    recall_list = [_.get_result() for _ in recall]
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


def user_discrete_feature_update(user_discrete_feature_list, user_ori_embed, ori_user_discrete_feature, current_dataloader, epoch_num, lr, vae_reg, vae_alpha, rec_model, my_vae):
    user_ori_embed_tensor = torch.tensor(user_ori_embed).to(device)
    user_discrete_feature_tensor = torch.tensor(user_discrete_feature_list).to(device)
    user_ori_user_discrete_tensor = torch.tensor(ori_user_discrete_feature).to(device)

    user_discrete_feature_tensor = torch.nn.Parameter(user_discrete_feature_tensor)

    optimizer = optim.SGD([user_discrete_feature_tensor], lr=lr)
    logsigmoid = torch.nn.LogSigmoid()
    mse_loss = torch.nn.MSELoss(reduction='sum')
    batch_input = user_ori_embed_tensor.unsqueeze(0)
    batch_ori_discrete_feature = user_ori_user_discrete_tensor.unsqueeze(0)     

    # print("------------------------------")
    loss_list = []

    for _ in range(epoch_num):
        epoch_loss = 0.
        # print(f"epoch:{_}")
        for batch_data in current_dataloader:
            batch_discrete_feature = user_discrete_feature_tensor.unsqueeze(0)

            out = my_vae.generate(batch_input, batch_ori_discrete_feature, batch_discrete_feature, vae_alpha)
            reg_loss = mse_loss(out, batch_input)
            out = out.squeeze(0)
            current_item_score_tensor = rec_model.get_item_score(out)           

            batch_user_list, batch_pos_item_list, batch_neg_item_list = batch_data
            batch_user_list = batch_user_list.to(device)
            batch_pos_item_list = batch_pos_item_list.to(device)
            batch_neg_item_list = batch_neg_item_list.to(device)    

            pos_score = current_item_score_tensor[batch_pos_item_list]
            neg_score = current_item_score_tensor[batch_neg_item_list]
            loss = - logsigmoid(pos_score - neg_score).sum() + vae_reg * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().cpu().item()

        epoch_loss = epoch_loss / len(current_dataloader.dataset)
        loss_list.append(epoch_loss)
        
    return user_discrete_feature_tensor.detach().cpu().to(torch.float32).tolist()

parser = argparse.ArgumentParser(description='test condition vae control')
parser.add_argument('--user', type=str, 
                    help='choose from train or test')
args = parser.parse_args()


setup_seed()

with open("./data/user_info.json", "r") as f:
    user_info = json.load(f)
with open("./data/item_info.json", "r") as f:
    item_info = json.load(f)
item_info = {int(k): v for k, v in item_info.items()}

user_num = 69878
item_num = 7254
tag_num = 19
rec_factor_num = 16
rec_num_layers = 3
rec_hidden_dim = rec_factor_num * (2 ** (rec_num_layers - 1))
use_gpu = True
device = torch.device('cuda') if (torch.cuda.is_available() and use_gpu) else torch.device('cpu')

rec_model = NCF(user_num, item_num, rec_factor_num, rec_num_layers)
rec_model.to(device)
pretrain_recmodel_path = "./checkpoint/0420-1758/NCF-epoch15.pkl"
if use_gpu:
    save_params = torch.load(pretrain_recmodel_path)
    rec_model.load_state_dict(save_params["model"])
else:
    save_params = torch.load(pretrain_recmodel_path, map_location='cpu')
    rec_model.load_state_dict(save_params["model"])
rec_model.eval()

topk = 1000
test_num = 1000

train_user_file = "./data/vae_train_user.txt"
test_user_file = "./data/vae_test_user.txt"
tag_into_file = "./data/tag_info.json"

train_user_dataset = UserDataset(train_user_file, tag_into_file, item_num, tag_num, rec_model, device)
test_user_dataset = UserDataset(test_user_file, tag_into_file, item_num, tag_num, rec_model, device)


if args.user == "train":
    user_list = train_user_dataset.user_list.copy()
    cur_user_dataset = train_user_dataset
elif args.user == "test":
    user_list = test_user_dataset.user_list.copy()
    cur_user_dataset = test_user_dataset
else:
    user_list = []
    print("Not support {}. Choose from train or test".format(args.user))

random.shuffle(user_list)
test_user_tag_list = []

for user in user_list[:test_num]: 
    tag = random.randint(0, tag_num-1)
    test_user_tag_list.append((user, tag))
print(f"len: {len(test_user_tag_list)}")
assert len(test_user_tag_list) == test_num

embed_size = rec_hidden_dim
feat_size = tag_num # tagvae_gaussian_dim 
h_dim = 128
gaussian_dim = 48
my_vae = condition_VAE5(embed_size, feat_size, h_dim, gaussian_dim, device)
my_vae = my_vae.to(device)
pretrain_vae_path = "./conditionvae_checkpoint/0421-1159/conditionvae5-epoch3649.pkl"
if use_gpu:
    save_params = torch.load(pretrain_vae_path)
    my_vae.load_state_dict(save_params["model"])
else:
    save_params = torch.load(pretrain_vae_path, map_location='cpu')
    my_vae.load_state_dict(save_params["model"])
my_vae.eval()
for para in my_vae.parameters():
    para.requires_grad = False

user_info = {int(k): [_[0] for _ in v] for k, v in user_info.items()}
rec_model.fix_item_para()


date_str = datetime.date.today().isoformat()
sys.stdout = Logger(f"conditionvae5-control-{args.user}-result-{date_str}-num-{test_num}-topk-{topk}-attscore.txt")
print(f"pretrain vae path {pretrain_vae_path}")
d_val_list = [-1., -0.8, -0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.]
d_val_len = len(d_val_list)
d_val_tensor = torch.tensor(d_val_list).to(device)
ap_diff_list = [[] for _ in range(d_val_len)]
ndcg_diff_list = [[] for _ in range(d_val_len)]
recall_diff_list = [[] for _ in range(d_val_len)]

random_ap_diff_list = [[] for _ in range(d_val_len)]
random_ndcg_diff_list = [[] for _ in range(d_val_len)]
random_recall_diff_list = [[] for _ in range(d_val_len)]

for user, tag in tqdm(test_user_tag_list): 

    user_idx = cur_user_dataset.name2id[user]
    user_ori_embed = cur_user_dataset.user_embeddings[user_idx]   

    user_discrete_feature = cur_user_dataset.user_discrete_features[user_idx] 

    ori_user_discrete_feature = user_discrete_feature.copy()    

    user_ori_embed_tensor = torch.tensor(user_ori_embed).to(device)
    user_discrete_feature_tensor = torch.tensor(user_discrete_feature).to(device)
    user_ori_user_discrete_tensor = torch.tensor(ori_user_discrete_feature).to(device) 

    with torch.no_grad():
        origin_item_score_tensor = rec_model.get_item_score(user_ori_embed_tensor)

    batch_input = user_ori_embed_tensor.repeat(d_val_len, 1)
    batch_discrete_feature = user_discrete_feature_tensor.repeat(d_val_len, 1)
    batch_ori_discrete_feature = user_ori_user_discrete_tensor.repeat(d_val_len, 1)
    batch_discrete_feature[:,tag] += d_val_tensor

    with torch.no_grad():
        out, mus_logs, z = my_vae.generate(batch_input, batch_ori_discrete_feature, batch_discrete_feature)
        current_item_score_tensor = rec_model.get_batch_item_score(out)    

    test_item_list = list(train_user_dataset.tag_info[tag])
    for idx in range(d_val_len):
        ap_diff, ndcg_diff, recall_diff = eval_rank_attribute(origin_item_score_tensor, current_item_score_tensor[idx], user, test_item_list, topk)
        ap_diff_list[idx].append(ap_diff)
        ndcg_diff_list[idx].append(ndcg_diff)
        recall_diff_list[idx].append(recall_diff)

    random_tag = random.randint(0, tag_num - 1)
    while random_tag == tag:
        random_tag = random.randint(0, tag_num - 1)
    test_item_list = list(train_user_dataset.tag_info[random_tag])
    for idx in range(d_val_len):
        ap_diff, ndcg_diff, recall_diff = eval_rank_attribute(origin_item_score_tensor, current_item_score_tensor[idx], user, test_item_list, topk)
        random_ap_diff_list[idx].append(ap_diff)
        random_ndcg_diff_list[idx].append(ndcg_diff)
        random_recall_diff_list[idx].append(recall_diff)    

ap_result_list = [cal_mean_and_var(_) for _ in ap_diff_list]
ndcg_result_list = [cal_mean_and_var(_) for _ in ndcg_diff_list]
recall_result_list = [cal_mean_and_var(_) for _ in recall_diff_list]

print('d_val ' + ','.join([str(_) for _ in d_val_list]))
print('ap ' + ','.join([str(_) for _ in ap_result_list]))
print('ndcg ' + ','.join([str(_) for _ in ndcg_result_list]))
print('recall ' + ','.join([str(_) for _ in recall_result_list]))


random_ap_result_list = [cal_mean_and_var(_) for _ in random_ap_diff_list]
random_ndcg_result_list = [cal_mean_and_var(_) for _ in random_ndcg_diff_list]
random_recall_result_list = [cal_mean_and_var(_) for _ in random_recall_diff_list]

print('random_ap ' + ','.join([str(_) for _ in random_ap_result_list]))
print('random_ndcg ' + ','.join([str(_) for _ in random_ndcg_result_list]))
print('random_recall ' + ','.join([str(_) for _ in random_recall_result_list]))