import torch
import torch.nn as nn
import torch.optim as optim

class SimpleRec(nn.Module):
    def __init__(self, user_num, item_num, hidden_dim):
        super(SimpleRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_dim = hidden_dim
        self.user_embeddings = nn.Embedding(self.user_num, self.hidden_dim)
        self.item_embeddings = nn.Embedding(self.item_num, self.hidden_dim)
        # self.item_idx = torch.tensor([_ for _ in range(item_num)])
        self.init_para()

    def init_para(self):
        # nn.init.xavier_normal_(self.user_embeddings.weight.data)
        # nn.init.xavier_normal_(self.item_embeddings.weight.data)
        self.user_embeddings.weight.data.normal_(0, 0.01)
        self.item_embeddings.weight.data.normal_(0, 0.01)

    def get_user_item_score(self, user_list, item_list):
        # assert user_list.size() == item_list.size()
        user_vec = self.user_embeddings(user_list)
        item_vec = self.item_embeddings(item_list)
        return torch.sum(user_vec * item_vec, dim=-1)

    def fix_item_para(self):
        self.item_embeddings.weight.requires_grad = False

    def unfix_item_para(self):
        self.item_embeddings.weight.requires_grad = True

    def get_item_score(self, user_vec, item_list=None):
        if item_list is None:
            item_vec = self.item_embeddings.weight
        else:
            item_vec = self.item_embeddings(item_list)
        return torch.sum(user_vec * item_vec, dim=-1)

    def get_all_item_score(self, user_list):
        user_vec = self.user_embeddings(user_list)
        all_item_rec = self.item_embeddings.weight
        return torch.mm(user_vec, all_item_rec.T)

    def get_batch_item_score(self, user_vec):
        all_item_rec = self.item_embeddings.weight
        return torch.mm(user_vec, all_item_rec.T)

class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers):
        super(NCF, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.factor_num = factor_num
        self.num_layers = num_layers
        self.user_embeddings = nn.Embedding(self.user_num, self.factor_num * (2 ** (self.num_layers - 1)))
        self.item_embeddings = nn.Embedding(self.item_num, self.factor_num * (2 ** (self.num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(0.1))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)
        self.predict_layer = nn.Linear(factor_num, 1)

        self.init_para()

    def init_para(self):
        # nn.init.xavier_normal_(self.user_embeddings.weight.data)
        # nn.init.xavier_normal_(self.item_embeddings.weight.data)
        self.user_embeddings.weight.data.normal_(0, 0.01)
        self.item_embeddings.weight.data.normal_(0, 0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, 
                                a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()


    def forward(self, user, item):
        embed_user = self.user_embeddings(user)
        embed_item = self.item_embeddings(item)
        interaction = torch.cat((embed_user, embed_item), -1)
        output_MLP = self.MLP_layers(interaction)
        prediction = self.predict_layer(output_MLP)
        return prediction.squeeze(-1)

    def get_item_score(self, user_vec, item_list=None):
        if item_list is None:
            item_vec = self.item_embeddings.weight
        else:
            item_vec = self.item_embeddings(item_list)
        user_vec = user_vec.unsqueeze(0).expand_as(item_vec)
        interaction = torch.cat((user_vec, item_vec), -1)
        output_MLP = self.MLP_layers(interaction)
        prediction = self.predict_layer(output_MLP)
        return prediction.squeeze(-1)        

    def fix_item_para(self):
        self.item_embeddings.weight.requires_grad = False
        for para in self.MLP_layers:
            para.requires_grad = False
        # for para in self.predict_layer:
        #     para.requires_grad = False
        self.predict_layer.requires_grad = False

    def unfix_item_para(self):
        self.item_embeddings.weight.requires_grad = True
        for para in self.MLP_layers:
            para.requires_grad = True
        # for para in self.predict_layer:
        #     para.requires_grad = True
        self.predict_layer.requires_grad = True

    def get_batch_item_score(self, user_vec, item_list=None):
        if item_list is None:
            all_item_vec = self.item_embeddings.weight
        else:
            all_item_vec = self.item_embeddings(item_list)

        user_dim = user_vec.size()[0]
        item_dim = all_item_vec.size()[0]

        user_vec_ex = user_vec.unsqueeze(1).repeat(1, item_dim, 1)
        item_vec_ex = all_item_vec.unsqueeze(0).repeat(user_dim, 1, 1)
        interaction = torch.cat((user_vec_ex, item_vec_ex), -1)
        output_MLP = self.MLP_layers(interaction)
        prediction = self.predict_layer(output_MLP)
        return prediction.squeeze(-1)

    def get_batch_item_score2(self, user_list):
        all_item_vec = self.item_embeddings.weight
        user_vec = self.user_embeddings(user_list)
        # print(f"user_vec: {user_vec.size()}")
        # print(f"all_item_vec: {all_item_vec.size()}")

        user_dim = user_vec.size()[0]
        item_dim = all_item_vec.size()[0]

        user_vec_ex = user_vec.unsqueeze(1).repeat(1, item_dim, 1)
        item_vec_ex = all_item_vec.unsqueeze(0).repeat(user_dim, 1, 1)
        interaction = torch.cat((user_vec_ex, item_vec_ex), -1)
        output_MLP = self.MLP_layers(interaction)
        prediction = self.predict_layer(output_MLP)
        return prediction.squeeze(-1)

    def get_user_item_score(self, user_list, item_list):
        if user_list.size() == item_list.size():
            return self.forward(user_list, item_list)

        assert user_list.size()[0] == 1

        user_vec = self.user_embeddings(user_list)
        item_vec = self.item_embeddings(item_list)

        item_dim = item_vec.size()[0]
        user_vec_ex = user_vec.repeat(item_dim, 1)
        interaction = torch.cat((user_vec_ex, item_vec), -1)
        output_MLP = self.MLP_layers(interaction)
        prediction = self.predict_layer(output_MLP)
        return prediction.squeeze(-1)

    def get_multi_user_one_item(self, user_vec_list, item):
        item_vec = self.item_embeddings(item)
        item_vec_ex = item_vec.repeat(user_vec_list.size()[0], 1)
        interaction = torch.cat((user_vec_list, item_vec_ex), -1)
        output_MLP = self.MLP_layers(interaction)
        prediction = self.predict_layer(output_MLP)
        return prediction.squeeze(-1)