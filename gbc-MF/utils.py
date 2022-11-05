import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from collections import defaultdict
from copy import deepcopy

def calculate_recall(item_score_numpy, test_item_list, topk=None):
    label_test_item = np.zeros(item_score_numpy.shape)
    label_test_item[test_item_list] = 1.
    item_score_rank = np.argsort(item_score_numpy)[::-1]
    item_label = label_test_item[item_score_rank]
    if topk is None: topk=len(item_label)
    return np.sum(item_label[:topk])/ min(topk, len(test_item_list))

def calculate_average_precision_score(item_score_numpy, test_item_list, topk=None):
    label_test_item = np.zeros(item_score_numpy.shape, dtype=int)
    label_test_item[test_item_list] = 1
    item_score_rank = np.argsort(item_score_numpy)[::-1]
    item_label = label_test_item[item_score_rank]
    if topk is None: topk=len(item_label)
    ap = 0.
    prec = 0.
    for i, label in enumerate(item_label[:topk]):
        if label == 1:
            prec += 1.
            ap += prec / (i + 1.)
    ap /= min(topk, len(test_item_list))
    return ap

def calculate_ndcg_score(item_score_numpy, test_item_list, topk=None):
    label_test_item = np.zeros(item_score_numpy.shape)
    label_test_item[test_item_list] = 1
    ndcg = metrics.ndcg_score(label_test_item.reshape(1, -1), item_score_numpy.reshape(1, -1), k=topk)
    return ndcg

def calculate_target_item_rank(item_score_tensor, target_item_idx):
    score_values, score_indices = item_score_tensor.sort(descending=True)
    score_indices = score_indices.cpu().tolist()
    target_item_rank = None
    for rank, item_id in enumerate(score_indices):
        if target_item_idx == item_id:
            target_item_rank = rank
            break
    return target_item_rank

class TaskDataset(Dataset):
    def __init__(self, data_list):
        super(TaskDataset, self).__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def get_sample(self, sample_size):
        if sample_size > len(self):
            sample_size = len(self)
        sample_idx = random.sample(range(len(self)), sample_size)
        return [self.data_list[idx] for idx in sample_idx]

class PosBPRTaskDataset(Dataset):
    def __init__(self, user, pos_item_list, neg_item_list):
        super(PosBPRTaskDataset, self).__init__()
        self.user = user
        self.pos_item_list = list(pos_item_list)
        self.neg_item_list = list(neg_item_list)

    def __len__(self):
        return len(self.pos_item_list)

    def __getitem__(self, index):
        return self.user, self.pos_item_list[index], random.choice(self.neg_item_list)

    def get_sample(self, sample_size):
        if sample_size > len(self):
            sample_size = len(self)
        sample_idx = random.sample(range(len(self)), sample_size)
        return [self.__getitem__(idx) for idx in sample_idx]

class NegBPRTaskDataset(Dataset):
    def __init__(self, user, pos_item_list, neg_item_list):
        super(NegBPRTaskDataset, self).__init__()
        self.user = user
        self.pos_item_list = list(pos_item_list)
        self.neg_item_list = list(neg_item_list)

    def __len__(self):
        return len(self.neg_item_list)

    def __getitem__(self, index):
        return self.user, random.choice(self.pos_item_list), self.neg_item_list[index]

    def get_sample(self, sample_size):
        if sample_size > len(self):
            sample_size = len(self)
        sample_idx = random.sample(range(len(self)), sample_size)
        return [self.__getitem__(idx) for idx in sample_idx]

def build_loader(task_dataset, batch_size, shuffle=True, num_threads=0):
    return DataLoader(
        task_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_threads,
    )

# from https://github.com/moskomule/ewc.pytorch/blob/master/utils.py
class EWC(object):
    def __init__(self, model: nn.Module, hisdata_dataloder: DataLoader, device):

        self.model = deepcopy(model)
        self.hisdata_dataloder = hisdata_dataloder
        self.device = device

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        logsigmoid = nn.LogSigmoid()
        self.model.eval()
        for his_data in self.hisdata_dataloder:
            self.model.zero_grad()
            u_id, pos_i_id, neg_i_id = his_data
            u_id = u_id.to(self.device)
            pos_i_id = pos_i_id.to(self.device)
            neg_i_id = neg_i_id.to(self.device)
            pos_score = self.model.get_user_item_score(u_id, pos_i_id)
            neg_score = self.model.get_user_item_score(u_id, neg_i_id)
            loss = - logsigmoid(pos_score - neg_score).sum()
            loss.backward()

            for n, p in self.model.named_parameters():
                if n in precision_matrices.keys():
                    precision_matrices[n].data += p.grad.data ** 2 / len(self.hisdata_dataloder)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._precision_matrices.keys():
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss

def normal_train_one_epoch(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader, device):
    model.train()
    logsigmoid = nn.LogSigmoid()
    epoch_loss = 0
    for batch_data in data_loader:
        batch_user_list, batch_pos_item_list, batch_neg_item_list = batch_data
        batch_user_list = batch_user_list.to(device)
        batch_pos_item_list = batch_pos_item_list.to(device)
        batch_neg_item_list = batch_neg_item_list.to(device)

        pos_score = model.get_user_item_score(batch_user_list, batch_pos_item_list)
        neg_score = model.get_user_item_score(batch_user_list, batch_neg_item_list)
        loss = - logsigmoid(pos_score - neg_score).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().cpu().item()
    return epoch_loss / len(data_loader)

def ewc_train_one_epoch(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
              ewc: EWC, importance: float, device):
    model.train()
    logsigmoid = nn.LogSigmoid()
    epoch_loss = 0
    for batch_data in data_loader:
        batch_user_list, batch_pos_item_list, batch_neg_item_list = batch_data
        batch_user_list = batch_user_list.to(device)
        batch_pos_item_list = batch_pos_item_list.to(device)
        batch_neg_item_list = batch_neg_item_list.to(device)

        pos_score = model.get_user_item_score(batch_user_list, batch_pos_item_list)
        neg_score = model.get_user_item_score(batch_user_list, batch_neg_item_list)
        loss = - logsigmoid(pos_score - neg_score).sum() + importance * ewc.penalty(model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().cpu().item()
    return epoch_loss / len(data_loader)

def eval_rank_attribute(origin_item_score_tensor, current_item_score_tensor, test_user, test_item_list, topk=None):
    old_item_score_numpy = origin_item_score_tensor.detach().cpu().numpy()
    current_item_score_numpy = current_item_score_tensor.detach().cpu().numpy()
    old_ap = calculate_average_precision_score(old_item_score_numpy, test_item_list, topk)
    old_ndcg = calculate_ndcg_score(old_item_score_numpy, test_item_list, topk)
    old_recall = calculate_recall(old_item_score_numpy, test_item_list, topk)
    current_ap = calculate_average_precision_score(current_item_score_numpy, test_item_list, topk)
    current_ndcg = calculate_ndcg_score(current_item_score_numpy, test_item_list, topk)
    curent_recall = calculate_recall(current_item_score_numpy, test_item_list, topk)
    ap_diff = old_ap - current_ap
    ndcg_diff = old_ndcg - current_ndcg
    recall_diff = old_recall - curent_recall
    return ap_diff, ndcg_diff, recall_diff

def eval_target_item_rank(origin_item_score_tensor, current_item_score_tensor, test_user, target_item_idx):
    old_item_score_numpy = origin_item_score_tensor.detach().cpu().numpy()
    current_item_score_numpy = current_item_score_tensor.detach().cpu().numpy()
    old_rank = calculate_target_item_rank(origin_item_score_tensor, target_item_idx)
    current_rank = calculate_target_item_rank(current_item_score_tensor, target_item_idx)
    # print(f"old rank {old_rank}  current rank {current_rank}")
    rank_diff = old_rank - current_rank
    return rank_diff

def eval_origin_model_attribute(origin_item_score_tensor, test_item_list):
    old_item_score_numpy = origin_item_score_tensor.detach().cpu().numpy()
    old_ap = calculate_average_precision_score(old_item_score_numpy, test_item_list)
    old_ndcg = calculate_ndcg_score(old_item_score_numpy, test_item_list)
    return old_ap, old_ndcg

def eval_origin_model_item_rank(origin_item_score_tensor, target_item_idx):
    old_item_score_numpy = origin_item_score_tensor.detach().cpu().numpy()
    old_rank = calculate_target_item_rank(origin_item_score_tensor, target_item_idx)
    return old_rank    

class Explainer(object):
    def __init__(self, model, device, item_info, user_info, item_num, epochs, lr, batch_size, sample_size, topk=None):
        self.origin_model = model
        self.device = device
        self.item_info = {int(k): v for k, v in item_info.items()}
        # self.user_info = {int(k): [_[0] for _ in v] for k, v in user_info.items()}
        self.user_info = {int(k): [_ for _ in v] for k, v in user_info.items()}
        self.item_num = item_num
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.topk = topk

        self.att2item = defaultdict(list)
        for item_idx in self.item_info:
            for att_idx in self.item_info[item_idx]:
                self.att2item[att_idx].append(item_idx)

    def make_init_dataloader(self, neg_num=5):
        data_list = []
        for pos_i_id in self.user_info[self.user_id]:
            for _ in range(neg_num):
                neg_i_id = np.random.randint(self.item_num)
                while neg_i_id in self.user_info[self.user_id]:
                    neg_i_id = np.random.randint(self.item_num)
                data_list.append([self.user_id, pos_i_id, neg_i_id])
        self.old_tasks_dataloader = [build_loader(TaskDataset(data_list), self.batch_size)]

    def init_train_process(self, user_id, update_mode="EWC", importance=500, epsilon=0.1): # update_mode in ['EWC', 'normal', 'both']
        self.user_id = user_id
        self.update_mode = update_mode

        if self.update_mode == "EWC" or self.update_mode == 'both':
            self.ewc_model = deepcopy(self.origin_model)
            self.ewc_model.fix_item_para()
            self.ewc_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.ewc_model.parameters()), lr=self.lr)
            self.importance = importance

        if self.update_mode == "normal" or self.update_mode == 'both':
            self.normal_model = deepcopy(self.origin_model)
            self.normal_model.fix_item_para()
            self.normal_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.normal_model.parameters()), lr=self.lr)

        if self.update_mode == "fgsm" or self.update_mode == 'both':
            self.fgsm_model = deepcopy(self.origin_model)
            self.fgsm_model.fix_item_para()
            self.epsilon = epsilon

        self.old_tasks_dataloader = []
        self.make_init_dataloader()

        self.critique_attribute_list = []

    def get_current_item_score(self, model_type="EWC"): # model_type in ['EWC', 'normal']
        if model_type == "EWC":
            self.ewc_model.eval()
            user_list_tensor = torch.tensor(self.user_id).unsqueeze(0)
            item_list_tensor = torch.tensor([_ for _ in range(self.item_num)])
            user_list_tensor = user_list_tensor.to(self.device)
            item_list_tensor = item_list_tensor.to(self.device)
            with torch.no_grad():
                current_item_score_tensor = self.ewc_model.get_user_item_score(user_list_tensor, item_list_tensor)
            return current_item_score_tensor
        elif model_type == "normal":
            self.normal_model.eval()
            user_list_tensor = torch.tensor(self.user_id).unsqueeze(0)
            item_list_tensor = torch.tensor([_ for _ in range(self.item_num)])
            user_list_tensor = user_list_tensor.to(self.device)
            item_list_tensor = item_list_tensor.to(self.device)
            with torch.no_grad():
                current_item_score_tensor = self.normal_model.get_user_item_score(user_list_tensor, item_list_tensor)
            return current_item_score_tensor
        elif model_type == "fgsm":
            self.fgsm_model.eval()
            user_list_tensor = torch.tensor(self.user_id).unsqueeze(0)
            item_list_tensor = torch.tensor([_ for _ in range(self.item_num)])
            user_list_tensor = user_list_tensor.to(self.device)
            item_list_tensor = item_list_tensor.to(self.device)
            with torch.no_grad():
                current_item_score_tensor = self.fgsm_model.get_user_item_score(user_list_tensor, item_list_tensor)
            return current_item_score_tensor            
        else:
            return None            

    def get_origin_item_score(self):
        self.origin_model.eval()
        user_list_tensor = torch.tensor(self.user_id).unsqueeze(0)
        item_list_tensor = torch.tensor([_ for _ in range(self.item_num)])
        user_list_tensor = user_list_tensor.to(self.device)
        item_list_tensor = item_list_tensor.to(self.device)
        with torch.no_grad():
            origin_item_score_tensor = self.origin_model.get_user_item_score(user_list_tensor, item_list_tensor)
        return origin_item_score_tensor

    def make_critique_dataloder(self, critique_attribute, test_pos):
        if test_pos:
            pos_item_set = set(self.att2item[critique_attribute])
            neg_item_set = set([_ for _ in range(self.item_num)]) - pos_item_set - set(self.user_info[self.user_id])
            return build_loader(PosBPRTaskDataset(self.user_id, list(pos_item_set), list(neg_item_set)), self.batch_size)            
        else:
            pos_item_set = set(self.user_info[self.user_id])
            critique_item_set = set(self.att2item[critique_attribute])
            return build_loader(NegBPRTaskDataset(self.user_id, list(pos_item_set - critique_item_set), list(critique_item_set)), self.batch_size)


    def update_model(self, critique_attribute, test_pos, silence=True):
        current_dataloader = self.make_critique_dataloder(critique_attribute, test_pos)

        if self.update_mode == "EWC" or self.update_mode == "both":
            old_tasks = []
            for each_loader in self.old_tasks_dataloader:
                old_tasks = old_tasks + each_loader.dataset.get_sample(self.sample_size)
            old_tasks = random.sample(old_tasks, k=min(self.sample_size, len(old_tasks)))
            old_tasks_dataloader = build_loader(TaskDataset(old_tasks), 1)
            current_EWC = EWC(self.ewc_model, old_tasks_dataloader, self.device)

            if silence:
                cur_iter = range(self.epochs)
            else:
                cur_iter = tqdm(range(self.epochs), ncols=0)

            epoch_loss_list = []
            for _ in cur_iter:
                epoch_loss = ewc_train_one_epoch(self.ewc_model, self.ewc_optimizer, current_dataloader, \
                                                    current_EWC, self.importance, self.device)
                epoch_loss_list.append(epoch_loss)
            if not silence:
                print("ewc epoch_loss_list: ", epoch_loss_list)

        if self.update_mode == "normal" or self.update_mode == "both":
            if silence:
                cur_iter = range(self.epochs)
            else:
                cur_iter = tqdm(range(self.epochs), ncols=0)

            epoch_loss_list = []
            for _ in cur_iter:
                epoch_loss = normal_train_one_epoch(self.normal_model, self.normal_optimizer, current_dataloader, self.device)
                epoch_loss_list.append(epoch_loss)
            if not silence:
                print("normal epoch_loss_list: ", epoch_loss_list)

        if self.update_mode == "fgsm" or self.update_mode == "both":
            user_list_tensor = torch.tensor(self.user_id).unsqueeze(0)
            item_list_tensor = torch.tensor(self.att2item[critique_attribute])
            user_list_tensor = user_list_tensor.to(self.device)
            item_list_tensor = item_list_tensor.to(self.device)
            
            self.fgsm_model.eval()
            current_item_score_tensor = torch.mean(self.fgsm_model.get_user_item_score(user_list_tensor, item_list_tensor))   
            self.fgsm_model.zero_grad()         
            current_item_score_tensor.backward()
            data_grad = self.fgsm_model.user_embeddings.weight.grad[self.user_id]

            data_grad = data_grad / torch.abs(data_grad).max()
            self.fgsm_model.user_embeddings.weight.data[self.user_id] = \
                self.fgsm_model.user_embeddings.weight.data[self.user_id] + self.epsilon * data_grad            


        self.old_tasks_dataloader.append(current_dataloader)
        self.critique_attribute_list.append(critique_attribute)

    def evaluate_model(self, target_item):
        return_dict = {}

        if self.update_mode == 'EWC' or self.update_mode == 'both':
            origin_item_score_tensor = self.get_origin_item_score()
            current_item_score_tensor = self.get_current_item_score(model_type='EWC')   

            ap_diff_list = []
            ndcg_diff_list = []
            recall_diff_list = []
            for critique_attribute in self.critique_attribute_list:
                test_item_list = list(self.att2item[critique_attribute])
                ap_diff, ndcg_diff, recall_diff = eval_rank_attribute(origin_item_score_tensor, current_item_score_tensor, self.user_id, test_item_list, self.topk)
                ap_diff_list.append(ap_diff)
                ndcg_diff_list.append(ndcg_diff)  
                recall_diff_list.append(recall_diff)  
            rank_diff = eval_target_item_rank(origin_item_score_tensor, current_item_score_tensor, self.user_id, target_item)

            return_dict['EWC'] = (ap_diff_list, ndcg_diff_list, recall_diff_list, rank_diff)

        if self.update_mode == 'normal' or self.update_mode == 'both':
            origin_item_score_tensor = self.get_origin_item_score()
            current_item_score_tensor = self.get_current_item_score(model_type='normal')   

            ap_diff_list = []
            ndcg_diff_list = []
            recall_diff_list = []
            for critique_attribute in self.critique_attribute_list:
                test_item_list = list(self.att2item[critique_attribute])
                ap_diff, ndcg_diff, recall_diff = eval_rank_attribute(origin_item_score_tensor, current_item_score_tensor, self.user_id, test_item_list, self.topk)
                ap_diff_list.append(ap_diff)
                ndcg_diff_list.append(ndcg_diff)    
                recall_diff_list.append(recall_diff)
            rank_diff = eval_target_item_rank(origin_item_score_tensor, current_item_score_tensor, self.user_id, target_item)

            return_dict['normal'] = (ap_diff_list, ndcg_diff_list, recall_diff_list, rank_diff)

        if self.update_mode == "fgsm" or self.update_mode == "both":
            origin_item_score_tensor = self.get_origin_item_score()
            current_item_score_tensor = self.get_current_item_score(model_type='fgsm')   

            ap_diff_list = []
            ndcg_diff_list = []
            recall_diff_list = []
            for critique_attribute in self.critique_attribute_list:
                test_item_list = list(self.att2item[critique_attribute])
                ap_diff, ndcg_diff, recall_diff = eval_rank_attribute(origin_item_score_tensor, current_item_score_tensor, self.user_id, test_item_list, self.topk)
                ap_diff_list.append(ap_diff)
                ndcg_diff_list.append(ndcg_diff)   
                recall_diff_list.append(recall_diff) 
            rank_diff = eval_target_item_rank(origin_item_score_tensor, current_item_score_tensor, self.user_id, target_item)

            return_dict['fgsm'] = (ap_diff_list, ndcg_diff_list, recall_diff_list, rank_diff)

        return return_dict

    def evaluate_origin_model(self, target_item):
        origin_item_score_tensor = self.get_origin_item_score()
        ap_list = []
        ndcg_list = []
        for critique_attribute in self.critique_attribute_list:
            test_item_list = list(self.att2item[critique_attribute])
            ap, ndcg = eval_origin_model_attribute(origin_item_score_tensor, test_item_list)
            ap_list.append(ap)
            ndcg_list.append(ndcg)
        item_rank = eval_origin_model_item_rank(origin_item_score_tensor, target_item)
        return ap_list, ndcg_list, item_rank

    def get_some_info(self):
        pos_item_num_list = []
        neg_item_num_list = []
        user_train_set = set(self.user_info[self.user_id])
        for critique_attribute in self.critique_attribute_list:
            neg_item_set = set(self.att2item[critique_attribute])
            pos_item_num_list.append(len(user_train_set - neg_item_set))
            neg_item_num_list.append(len(neg_item_set))
        return len(user_train_set), pos_item_num_list, neg_item_num_list

    def save_current_model(self, save_file_path, model_type='EWC'): # model_type in ['EWC', 'normal']
        save_model = None
        if model_type == 'EWC':
            save_model = self.ewc_model
        if model_type == 'normal':
            save_model = self.normal_model
        if model_type == 'fgsm':
            save_model = self.fgsm_model
        save_params = {
            "model": save_model.state_dict(),
        }
        torch.save(save_params, save_file_path)