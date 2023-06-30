import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20
import time
from tqdm import tqdm
from torch import optim
class LightGCN(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LightGCN, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['LightGCN'])
        self.n_layers = int(args['-n_layer'])
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.user_dataset = []
        self.item_dataset = []

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model()

            self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def TransR_train(self,dataset):
        Recmodel = self.model.to(self.device)
        Recmodel.train()
        opt = optim.Adam(Recmodel.parameters(), lr=float(self.config['lr']))

        kgloader = dataset
        trans_loss = 0.0
        for data in tqdm(kgloader, total=len(kgloader)):
            heads = data[0].to(self.device)
            relations = data[1].to(self.device)
            pos_tails = data[2].to(self.device)
            neg_tails = data[3].to(self.device)
            kg_batch_loss = Recmodel.calc_kg_loss_transE(heads, relations, pos_tails, neg_tails)

            trans_loss += kg_batch_loss / len(kgloader)
            opt.zero_grad()
            kg_batch_loss.backward()
            opt.step()



    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings

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


