import os.path
from os import remove
from re import split
from torch.utils.data import Dataset, DataLoader
from os.path import join
import pandas as pd
import collections
import torch
import random
from tqdm import tqdm
import json

class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def write_file(dir, file, content, op='w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir + file, op) as f:
            f.writelines(content)

    @staticmethod
    def delete_file(file_path):
        if os.path.exists(file_path):
            remove(file_path)

    @staticmethod
    def load_data_set(file, dtype):
        data = []
        if dtype == 'graph':
            with open(file) as f:
                for line in f:
                    items = split(' ', line.strip())
                    user_id = items[0]
                    item_id = items[1]
                    weight = items[2]
                    data.append([user_id, item_id, float(weight)])

        if dtype == 'sequential':
            training_data, test_data = [], []
            with open(file) as f:
                for line in f:
                    items = split(':', line.strip())
                    user_id = items[0]
                    seq = items[1].strip().split()
                    training_data.append(seq[:-1])
                    test_data.append(seq[-1])
                data = (training_data, test_data)
        return data

    @staticmethod
    def load_user_list(file):
        user_list = []
        print('loading user List...')
        with open(file) as f:
            for line in f:
                user_list.append(line.strip().split()[0])
        return user_list

    @staticmethod
    def load_social_data(file):
        social_data = []
        print('loading social data...')
        with open(file) as f:
            for line in f:
                items = split(' ', line.strip())
                user1 = items[0]
                user2 = items[1]
                if len(items) < 3:
                    weight = 1
                else:
                    weight = float(items[2])
                social_data.append([user1, user2, weight])
        return social_data


'''kg数据导入处理'''
class KGDataset(Dataset):
    def __init__(self, conf):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.entity_num_per_item = conf['entity_num_per_item']

        DATA_PATH = conf['DATA_PATH']
        dataset = conf['dataset']
        kg_path = join(DATA_PATH, dataset, "kg_final.txt")
        kg_data = pd.read_csv(kg_path, sep=' ', names=['h', 'r', 't'], engine='python')
        self.kg_data = kg_data.drop_duplicates()
        self.kg_dict, self.heads = self.generate_kg_data(kg_data=self.kg_data)
        self.item_net_path = join(DATA_PATH, dataset)

    def generate_kg_data(self, kg_data):
        # construct kg dict
        kg_dict = collections.defaultdict(list)
        print("load kg graph")
        for row in tqdm(kg_data.iterrows(), total=len(kg_data)):
            h, r, t = row[1]
            kg_dict[h].append((r, t))
        heads = list(kg_dict.keys())
        return kg_dict, heads

    '''添加只读属性，用于获取实体和关系长度'''
    @property
    def entity_count(self):
        # start from zero
        return max(max(self.kg_data['h']), max(self.kg_data['t'])) + 1

    @property
    def relation_count(self):
        return max(self.kg_data['r']) + 2

    def get_kg_dict(self, item_num):
        print("set item_entity kg dict")
        entity_num = int(self.entity_num_per_item)
        i2es = dict()
        i2rs = dict()
        i2es_file = join(self.item_net_path,'i2es.pt')
        i2rs_file = join(self.item_net_path,'i2rs.pt')
        if os.path.exists(i2es_file) and os.path.exists(i2rs_file):
            i2es = torch.load(i2es_file)
            i2rs = torch.load(i2rs_file)

        else:
            for item in tqdm(range(item_num)):
                rts = self.kg_dict.get(item, False)
                if rts:
                    tails = list(map(lambda x: x[1], rts))
                    relations = list(map(lambda x: x[0], rts))
                    if (len(tails) > entity_num):
                        i2es[item] = torch.IntTensor(tails).to(self.device)[:entity_num]
                        i2rs[item] = torch.IntTensor(relations).to(self.device)[:entity_num]
                    else:
                        # last embedding pos as padding idx
                        tails.extend([self.entity_count] * (entity_num - len(tails)))
                        relations.extend([self.relation_count] * (entity_num - len(relations)))
                        i2es[item] = torch.IntTensor(tails).to(self.device)
                        i2rs[item] = torch.IntTensor(relations).to(self.device)
                else:
                    i2es[item] = torch.IntTensor([self.entity_count] * entity_num).to(self.device)
                    i2rs[item] = torch.IntTensor([self.relation_count] * entity_num).to(self.device)

            torch.save(i2es,i2es_file)
            torch.save(i2rs, i2rs_file)
        return i2es, i2rs

    def __len__(self):
        return len(self.kg_dict)

    def __getitem__(self, index):
        head = self.heads[index]
        relation, pos_tail = random.choice(self.kg_dict[head])
        while True:
            neg_head = random.choice(self.heads)
            neg_tail = random.choice(self.kg_dict[neg_head])[1]
            if (relation, neg_tail) in self.kg_dict[head]:
                continue
            else:
                break
        return head, relation, pos_tail, neg_tail



'''kg数据导入处理'''
class KGDataset2(Dataset):
    def __init__(self, data):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kg_data = data
        self.kg_dict, self.heads = self.generate_kg_data(kg_data=self.kg_data)

    def generate_kg_data(self, kg_data):
        # construct kg dict
        kg_dict = collections.defaultdict(list)
        print("load kg graph")
        for row in tqdm(kg_data.iterrows(), total=len(kg_data)):
            h, r, t = row[1]
            kg_dict[h].append((r, t))
        heads = list(kg_dict.keys())
        return kg_dict, heads

    @property
    def entity_count(self):
        # start from zero
        return max(max(self.kg_data['h']), max(self.kg_data['t'])) + 1

    @property
    def relation_count(self):
        return max(self.kg_data['r']) + 2

    def __len__(self):
        return len(self.kg_dict)

    def __getitem__(self, index):
        head = self.heads[index]
        relation, pos_tail = random.choice(self.kg_dict[head])
        while True:
            neg_head = random.choice(self.heads)
            neg_tail = random.choice(self.kg_dict[neg_head])[1]
            if (relation, neg_tail) in self.kg_dict[head]:
                continue
            else:
                break
        return head, relation, pos_tail, neg_tail


