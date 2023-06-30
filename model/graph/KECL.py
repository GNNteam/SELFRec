import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.loader import KGDataset
from tqdm import tqdm
from torch import optim
from model.graph.GAT import GAT
from torch.optim import optimizer, lr_scheduler


import time
# Paper: KECL - Towards Extremely Simple Graph Contrastive Learning for Recommendation

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class KECL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(KECL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['KECL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.temp = float(args['-tau'])
        self.n_layers = int(args['-n_layer'])
        self.layer_cl = int(args['-l*'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kg_dataset = KGDataset(self.config)
        self.model = KECL_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.layer_cl, self.kg_dataset)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb = model(True)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, pos_idx], rec_user_emb, cl_user_emb, rec_item_emb,
                                                          cl_item_emb)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx, user_view1, user_view2, item_view1, item_view2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)

        return user_cl_loss + item_cl_loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class KECL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, layer_cl, kg_dataset):
        super(KECL_Encoder, self).__init__()
        self.kg_dataset = kg_dataset
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.layer_cl = layer_cl
        self.norm_adj = data.norm_adj

        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.__init_kg_weight()
        self.embedding_dict = self._init_model()
        self.kg_dict, self.item2relations = self.kg_dataset.get_kg_dict(self.data.item_num)
        self.gat = GAT(self.emb_size, self.emb_size, dropout=0, alpha=0.2).train()

    def __init_kg_weight(self):
        self.num_entities = self.kg_dataset.entity_count
        self.num_relations = self.kg_dataset.relation_count
        self.W_R = nn.Parameter(torch.Tensor(self.num_relations, self.emb_size, self.emb_size))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))
        # self.kg_dict, self.item2relations = self.kg_dataset.get_kg_dict(self.data.item_num)
        self.embedding_entity = torch.nn.Embedding(
            num_embeddings=self.num_entities + 1, embedding_dim=self.emb_size)
        self.embedding_relation = torch.nn.Embedding(
            num_embeddings=self.num_relations + 1, embedding_dim=self.emb_size)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })

        return embedding_dict

    def calc_kg_loss_transE(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.embedding_relation(r)  # (kg_batch_size, relation_dim)
        h_embed = self.embedding_entity(h)  # (kg_batch_size, entity_dim)
        pos_t_embed = self.embedding_entity(pos_t)  # (kg_batch_size, entity_dim)
        neg_t_embed = self.embedding_entity(neg_t)  # (kg_batch_size, entity_dim)
        # Equation (1)
        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)  # (kg_batch_size)
        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(
            neg_t_embed)
        # # TODO: optimize L2 weight
        loss = kg_loss + 1e-3 * l2_loss
        # loss = kg_loss
        return loss

    def cal_item_embedding_from_kg(self):
        kg = self.kg_dict
        item_embs = self.embedding_dict['item_emb'][torch.tensor(list(kg.keys()))].cuda()  # item_num, emb_dim
        item_entities = torch.stack(list(kg.values()))  # item_num, entity_num_each
        item_relations = torch.stack(list(self.item2relations.values()))

        entity_embs = self.embedding_entity(item_entities)  # item_num, entity_num_each, emb_dim
        relation_embs = self.embedding_relation(item_relations)  # item_num, entity_num_each, emb_dim
        # w_r = self.W_R[relation_embs] # item_num, entity_num_each, emb_dim, emb_dim
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(item_entities),
                                   torch.zeros_like(item_entities)).float()
        return self.gat.forward_relation(item_embs, entity_embs, relation_embs, padding_mask)

    def updata_items_emb(self):
        # self.embedding_dict['item_emb'][:self.test] = self.cal_item_embedding_from_kg()
        pass

    def forward(self, perturbed=False):
        users_emb = self.embedding_dict['user_emb']
        items_emb = self.embedding_dict['item_emb']
        # ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        ego_embeddings = torch.cat([users_emb, items_emb])
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                items_emb2 = torch.sigmoid(self.cal_item_embedding_from_kg())
                users_emb2 = torch.rand_like(users_emb).cuda()
                random_noise = torch.cat([users_emb2, items_emb2])
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
                # ego_embeddings += random_noise * self.eps
            all_embeddings.append(ego_embeddings)
            if k == self.layer_cl - 1:
                all_embeddings_cl = ego_embeddings
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings,
                                                               [self.data.user_num, self.data.item_num])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl,
                                                                     [self.data.user_num, self.data.item_num])
        if perturbed:
            return user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings
