import numpy as np
from collections import defaultdict
from data.data import Data
from data.graph import Graph
import scipy.sparse as sp
import pickle
from collections import Counter


class Interaction(Data, Graph):
    def __init__(self, conf, training, test):
        Graph.__init__(self)
        Data.__init__(self, conf, training, test)

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_item = set()

        self.__generate_set()
        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.__create_sparse_interaction_matrix()
        self.test_set_group = []
        self.test_set_cold = defaultdict(dict)
        self.test_user_cold = []
        # self._cold_user()
        # self.test_set_group = self._group_item_by_frequency()

        # popularity_user = {}
        # for u in self.user:
        #     popularity_user[self.user[u]] = len(self.training_set_u[u])
        # popularity_item = {}
        # for u in self.item:
        #     popularity_item[self.item[u]] = len(self.training_set_i[u])

    def _cold_user(self):
        frequency = {}
        for key in self.training_set_u.keys():
            frequency[key] = len(self.training_set_u[key])
        user_frequency = sorted(frequency.items(), key=lambda x: x[1])

        cold_user = [t[0] for t in user_frequency if t[1] < 15]
        for test_set in self.test_set:
            if test_set in cold_user:
                self.test_set_cold[test_set] = self.test_set[test_set]
                self.test_user_cold.append(test_set)


    def _group_item_by_frequency(self):
        frequency = {}
        for key in self.training_set_i.keys():
            frequency[key] = len(self.training_set_i[key])
        group_fre = sorted(frequency.items(), key=lambda x: x[1])
        fre_list = []
        for t in group_fre:
            fre_list.append(t[0])

        fre_list = [item for item in fre_list if item in self.test_set_item]
        filtered_test_data = [x for x in self.test_data if x[1] in fre_list]
        sort_test_data = sorted(filtered_test_data, key=lambda x: fre_list.index(x[1]))
        # sort_test_data = sorted(self.test_data, key=lambda x: fre_list.index(x[1]))

        # 将sort_test_data平均分成5组
        num_groups = 5
        group_size = len(sort_test_data) // num_groups
        remainder = len(sort_test_data) % num_groups
        group_list = []
        groups = []
        start_index = 0
        for i in range(num_groups):
            if i < remainder:
                end_index = start_index + group_size + 1
            else:
                end_index = start_index + group_size

            groups.append(sort_test_data[start_index:end_index])
            start_index = end_index

        for group in groups:
            temp_dic = defaultdict(dict)
            for entry in group:
                user, item, rating = entry
                temp_dic[user][item] = rating
            group_list.append(temp_dic)
        return group_list

    def __generate_set(self):
        for entry in self.training_data:
            user, item, rating = entry
            if user not in self.user:
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                # userList.append
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating
        for entry in self.test_data:
            user, item, rating = entry
            if user not in self.user:
                continue
            self.test_set[user][item] = rating
            self.test_set_item.add(item)

    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),
                                dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0] + adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])), shape=(n_nodes, n_nodes),
                                dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):
        """
        return a sparse adjacency matrix with the shape (user number, item number)
        """
        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num, self.item_num), dtype=np.float32)
        return interaction_mat

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):
        'whether user u rated item i'
        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_user(self, u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def contain_item(self, i):
        """whether item is in training set"""
        if i in self.item:
            return True
        else:
            return False

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        u = self.id2user[u]
        k, v = self.user_rated(u)
        vec = np.zeros(len(self.item))
        # print vec
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        i = self.id2item[i]
        k, v = self.item_rated(i)
        vec = np.zeros(len(self.user))
        # print vec
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.user:
            k, v = self.user_rated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]] = vec
        return m
