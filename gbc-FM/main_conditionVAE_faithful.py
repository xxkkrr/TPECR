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

        batch = 1024
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


def user_discrete_feature_update(user_discrete_feature_list, user_ori_embed, ori_user_discrete_feature, pre_user_ori_embed, current_dataloader, epoch_num, lr, vae_reg1, vae_reg2, vae_reg3, reg_update_step, vae_alpha, iter_num, rec_model, my_vae):
    user_ori_embed_tensor = torch.tensor(user_ori_embed).to(device)
    user_discrete_feature_tensor = torch.tensor(user_discrete_feature_list).to(device)
    user_ori_user_discrete_tensor = torch.tensor(ori_user_discrete_feature).to(device)
    pre_user_ori_embed = pre_user_ori_embed.to(device)
    pre_user_discrete_feature = user_discrete_feature_tensor.clone()

    user_discrete_feature_tensor = torch.nn.Parameter(user_discrete_feature_tensor)
    optimizer = optim.SGD([user_discrete_feature_tensor], lr=lr)
    logsigmoid = torch.nn.LogSigmoid()
    mse_loss = torch.nn.MSELoss(reduction='sum')
    batch_input = user_ori_embed_tensor.unsqueeze(0)
    batch_ori_discrete_feature = user_ori_user_discrete_tensor.unsqueeze(0)     

    loss_list = []

    for _ in range(epoch_num):
        epoch_loss = 0.
        # print(f"epoch:{_}")
        # print(f"1 user_discrete_feature_tensor {user_discrete_feature_tensor}")
        for batch_data in current_dataloader:
            batch_discrete_feature = user_discrete_feature_tensor.unsqueeze(0)

            out, mus_logs, z = my_vae.generate2(batch_input, batch_discrete_feature)
            reg_loss2 = mse_loss(out, pre_user_ori_embed.unsqueeze(0))
            reg_loss3 = mse_loss(batch_discrete_feature, pre_user_discrete_feature.unsqueeze(0))
            out = out.squeeze(0)
            current_item_score_tensor = rec_model.get_item_score(out)     
            current_tag_score = rec_model.get_att_score(out)
            reg_loss1 = mse_loss(batch_discrete_feature, current_tag_score.unsqueeze(0))

            batch_user_list, batch_pos_item_list, batch_neg_item_list = batch_data
            batch_user_list = batch_user_list.to(device)
            batch_pos_item_list = batch_pos_item_list.to(device)
            batch_neg_item_list = batch_neg_item_list.to(device)    

            pos_score = current_item_score_tensor[batch_pos_item_list]
            neg_score = current_item_score_tensor[batch_neg_item_list]
            train_loss = - logsigmoid(pos_score - neg_score).sum()

            loss = train_loss + vae_reg1 * reg_loss1 + vae_reg2 * reg_loss2 + vae_reg3 * reg_loss3

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

    with torch.no_grad():
        out, mus_logs, z = my_vae.generate2(batch_input, batch_discrete_feature)

    return user_discrete_feature_tensor.detach().cpu().to(torch.float32).tolist(), out.detach()

def user_discrete_feature_update2(user_discrete_feature_list, user_ori_embed, ori_user_discrete_feature, iter_num, rec_model, my_vae):
    user_ori_embed_tensor = torch.tensor(user_ori_embed).to(device)
    user_discrete_feature_tensor = torch.tensor(user_discrete_feature_list).to(device)
    user_ori_user_discrete_tensor = torch.tensor(ori_user_discrete_feature).to(device)

    tag_matrix = train_user_dataset.tag_matrix.to(device)
    tag_len = train_user_dataset.tag_len.to(device)

    with torch.no_grad():
        for _ in range(iter_num):
            batch_input = user_ori_embed_tensor.unsqueeze(0)
            batch_ori_discrete_feature = user_ori_user_discrete_tensor.unsqueeze(0)     
            batch_discrete_feature = user_discrete_feature_tensor.unsqueeze(0)
            out, mus_logs, z = my_vae.generate2(batch_input, batch_discrete_feature)
            out = out.squeeze(0)
            current_item_score_tensor = rec_model.get_item_score(out)     
            current_tag_score = torch.mm(current_item_score_tensor.unsqueeze(0), tag_matrix)
            current_tag_score = current_tag_score / tag_len
            user_discrete_feature_tensor = current_tag_score.squeeze(0)

        batch_discrete_feature = user_discrete_feature_tensor.unsqueeze(0)
        out, mus_logs, z = my_vae.generate2(batch_input, batch_discrete_feature)

    return user_discrete_feature_tensor.detach().cpu().to(torch.float32).tolist(), out


def cal_losses(user_discrete_feature_list_gt, user_discrete_feature_list, user_ori_embed, ori_user_discrete_feature, current_dataloader, rec_model, my_vae, vae_alpha):
    user_ori_embed_tensor = torch.tensor(user_ori_embed).to(device)
    user_ori_user_discrete_tensor = torch.tensor(ori_user_discrete_feature).to(device)    
    batch_input = user_ori_embed_tensor.unsqueeze(0)
    batch_ori_discrete_feature = user_ori_user_discrete_tensor.unsqueeze(0)    

    user_discrete_feature_tensor = torch.tensor(user_discrete_feature_list).to(device)
    user_discrete_feature_tensor_gt = torch.tensor(user_discrete_feature_list_gt).to(device)

    loss_gt = torch.tensor(0.).to(device)
    loss = torch.tensor(0.).to(device)
    mse_loss = torch.nn.MSELoss(reduction='sum')   
    logsigmoid = torch.nn.LogSigmoid() 

    batch_discrete_feature = user_discrete_feature_tensor.unsqueeze(0)
    out, embed_mus_logs, feat_out, feat_mus_logs = my_vae(batch_input, batch_discrete_feature)
    batch_discrete_feature_gt = user_discrete_feature_tensor_gt.unsqueeze(0)
    out_gt, embed_mus_logs, feat_out, feat_mus_logs = my_vae(batch_input, batch_discrete_feature_gt)

    reg_loss2 = mse_loss(out, batch_input)
    reg_loss2_gt = mse_loss(out_gt, batch_input)

    tag_matrix = train_user_dataset.tag_matrix.to(device)
    tag_len = train_user_dataset.tag_len.to(device)

    out = out.squeeze(0)
    current_item_score_tensor = rec_model.get_item_score(out)     
    current_tag_score = torch.mm(current_item_score_tensor.unsqueeze(0), tag_matrix)
    current_tag_score = current_tag_score / tag_len
    reg_loss1 = mse_loss(current_tag_score, batch_discrete_feature)

    out_gt = out_gt.squeeze(0)
    current_item_score_tensor_gt = rec_model.get_item_score(out_gt)     
    current_tag_score_gt = torch.mm(current_item_score_tensor_gt.unsqueeze(0), tag_matrix)
    current_tag_score_gt = current_tag_score_gt / tag_len
    reg_loss1_gt = mse_loss(current_tag_score_gt, batch_discrete_feature_gt)


    for batch_data in current_dataloader:
        batch_user_list, batch_pos_item_list, batch_neg_item_list = batch_data
        batch_user_list = batch_user_list.to(device)
        batch_pos_item_list = batch_pos_item_list.to(device)
        batch_neg_item_list = batch_neg_item_list.to(device)    

        pos_score = current_item_score_tensor[batch_pos_item_list]
        neg_score = current_item_score_tensor[batch_neg_item_list]
        loss += - logsigmoid(pos_score - neg_score).sum()

        pos_score = current_item_score_tensor_gt[batch_pos_item_list]
        neg_score = current_item_score_tensor_gt[batch_neg_item_list]
        loss_gt += - logsigmoid(pos_score - neg_score).sum()

    print(f"loss: {loss}, reg_loss1: {reg_loss1}, reg_loss2: {reg_loss2}")
    print(f"loss_gt: {loss_gt}, reg_loss1_gt: {reg_loss1_gt}, reg_loss2: {reg_loss2_gt}")

def MAPE(predict, target):
    loss = (predict - target).abs() / (target.abs() + 1e-8)
    return loss.mean()


setup_seed()

with open("./data/user_info.json", "r") as f:
    user_info = json.load(f)
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

topk = 1000
tag_sample_num = 1
test_num = 1000
test_pos = True
user_mode = "train"

train_user_file = "./data/vae_train_user.txt"
test_user_file = "./data/vae_test_user.txt"
tag_into_file = "./data/tag_info.json"

if user_mode == "train":
    test_file = "./data/vae_test_user_item.txt"
    train_user_dataset = UserDataset(train_user_file, tag_into_file, item_num, tag_num, rec_model, device)
else:
    test_file = "./data/vae_test_user_item_unseen.txt"
    train_user_dataset = UserDataset(test_user_file, tag_into_file, item_num, tag_num, rec_model, device)
test_user_item_pair = load_user_item_pair(test_file)
random.shuffle(test_user_item_pair)

all_tag_diff_list = []
all_test_user_item = []

for user, item in test_user_item_pair: 

    user_idx = train_user_dataset.name2id[user]
    user_ori_embed = train_user_dataset.user_embeddings[user_idx]   

    user_discrete_feature = train_user_dataset.user_discrete_features[user_idx] 

    ori_user_discrete_feature = user_discrete_feature.copy()    

    user_01_feature = train_user_dataset.user_01_features[user_idx] 

    item_discrete_feature = [0] * tag_num
    for tag in item_info[item]:
        item_discrete_feature[tag] = 1  

    user_discrete_feature_sorted_index = [_[0] for _ in sorted(enumerate(user_discrete_feature), key=lambda x:x[1])] 

    tag_diff_list = []
    if test_pos:
        for idx in user_discrete_feature_sorted_index:
            if user_01_feature[idx] < item_discrete_feature[idx]:
                tag_diff_list.append((idx, test_pos))
    else:
        for idx in user_discrete_feature_sorted_index[::-1]:
            if user_01_feature[idx] > item_discrete_feature[idx]:
                tag_diff_list.append((idx, test_pos))     

    if len(tag_diff_list) < tag_sample_num : 
        continue    

    flag = True
    for tag, test_pos in tag_diff_list:
        pos_item_set = set(user_info[str(user)])
        critique_item_set = set(train_user_dataset.tag_info[tag])
        if len(pos_item_set - critique_item_set) == 0:
            flag = False
            break
    if not flag and not test_pos:
        continue

    tag_diff_list = tag_diff_list[:tag_sample_num]
    all_tag_diff_list.append(tag_diff_list)
    all_test_user_item.append((user, item))

    if len(all_tag_diff_list) == test_num:
        break

assert len(all_tag_diff_list) == len(all_test_user_item)

epochs = 10
lr = 2e-5
batch_size = 128
sample_size = 200
importance = 10
my_explainer = Explainer(rec_model, device, item_info, user_info, item_num, epochs, lr, batch_size, sample_size, topk)

normal_ap = [[] for _ in range(tag_sample_num)]
normal_ndcg = [[] for _ in range(tag_sample_num)]
normal_recall = [[] for _ in range(tag_sample_num)]
normal_rank = [[] for _ in range(tag_sample_num)]


embed_size = rec_hidden_dim
feat_size = tag_num # tagvae_gaussian_dim 
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

user_info = {int(k): [_ for _ in v] for k, v in user_info.items()}


vae_ap = [[] for _ in range(tag_sample_num)]
vae_ndcg = [[] for _ in range(tag_sample_num)]
vae_recall = [[] for _ in range(tag_sample_num)]
vae_rank = [[] for _ in range(tag_sample_num)]

rec_model.fix_item_para()

vae_epochs = 10
vae_lr = 1e-2
vae_batch_size = 256
# clip_norm = 5.0
vae_alpha = None
vae_reg1 = 0.
vae_reg2 = 15.
vae_reg3 = 0.
reg_update_step = None

iter_num = None

ori_gt_loss = []
gt_loss = []
gt_mse_loss = []
mse_loss = nn.MSELoss(reduction='sum')

mse_list = []
mape_list = []
ndcg_list = []

e_mse_list = []
e_mape_list = []

date_str = datetime.date.today().isoformat()
sys.stdout = Logger(f"conditionvae9-faithful-result-{date_str}-num-{test_num}-topk-{topk}-att-{tag_sample_num}-attscore-onetag-pos-{test_pos}-user-{user_mode}-vae_reg1-{vae_reg1}-vae_reg2-{vae_reg2}-normreg.txt")
print(f"baseline epochs:{epochs} lr:{lr} batch_size:{batch_size} sample_size:{sample_size} importance:{importance}")
print(f"cvae epochs:{vae_epochs} lr:{vae_lr} batch_size:{vae_batch_size} reg1:{vae_reg1} reg2:{vae_reg2} reg3:{vae_reg3} alpha:{vae_alpha} reg_update_step:{reg_update_step} iter_num:{iter_num}")
print(f"pretrain vae path {pretrain_vae_path} test_pos {test_pos} user_mode {user_mode}")

all_user_list = list(range(len(train_user_dataset.user_embeddings)))
user1_list = random.sample(all_user_list, 5000)
user2_list = random.sample(all_user_list, 5000)
user_pair_list = list(zip(user1_list, user2_list))

print(user_pair_list[:10])
for user_idx, new_user_idx in user_pair_list:
    if user_idx == new_user_idx:
        continue
    user_ori_embed = train_user_dataset.user_embeddings[user_idx]
    user_discrete_feature = train_user_dataset.user_discrete_features[new_user_idx]
    gt_embed = train_user_dataset.user_embeddings[new_user_idx]

    user_ori_embed_tensor = torch.tensor(user_ori_embed).to(device)
    user_discrete_feature_tensor = torch.tensor(user_discrete_feature).to(device)

    batch_input = user_ori_embed_tensor.unsqueeze(0)
    batch_discrete_feature = user_discrete_feature_tensor.unsqueeze(0)
    out, mus_logs, z = my_vae.generate2(batch_input, batch_discrete_feature)
    out = out.squeeze(0)
    current_tag_score = rec_model.get_att_score(out)
    current_tag_score = current_tag_score.detach().cpu().tolist()

    mse_error = metrics.mean_squared_error(user_discrete_feature, current_tag_score)
    mse_list.append(mse_error)
    mape_error = metrics.mean_absolute_percentage_error(user_discrete_feature, current_tag_score)
    mape_list.append(mape_error)
    ndcg_score = metrics.ndcg_score(np.asarray([user_discrete_feature])-min(user_discrete_feature), np.asarray([current_tag_score]))
    ndcg_list.append(ndcg_score)

    out = out.detach().cpu().tolist()
    e_mse = metrics.mean_squared_error(gt_embed, out)
    e_mse_list.append(e_mse)
    e_mape = metrics.mean_absolute_percentage_error(gt_embed, out)
    e_mape_list.append(e_mape)

print(f"mse {cal_mean_and_var(mse_list)}")
print(f"mape {cal_mean_and_var(mape_list)}")
print(f"ndcg {cal_mean_and_var(ndcg_list)}")
print(f"e_mse {cal_mean_and_var(e_mse_list)}")
print(f"e_mape {cal_mean_and_var(e_mape_list)}")