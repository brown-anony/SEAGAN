import os
import json
import torch
from torch_geometric.io import read_txt_array
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import sort_edge_index
import numpy as np
import torch.nn as nn
import tqdm
import torch.nn.functional as F
from torch.autograd import Variable

class DBP15K(InMemoryDataset):
    # def __init__(self, root, pair, KG_num=1, rate=0.02, rate2=0.03,seed=1):
    def __init__(self, root, pair, KG_num=1, rate=0.2, rate2=0.35,seed=1):
        self.pair = pair
        self.KG_num = KG_num
        self.rate = rate
        self.rate2= rate2
        self.seed = seed
        torch.manual_seed(seed)
        super(DBP15K, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['zh_en', 'fr_en', 'ja_en']

    @property
    def processed_file_names(self):
        return '%s_%d_%.1f_%.1f_%d.pt' % (self.pair, self.KG_num, self.rate, self.rate2,self.seed)
    


    def process(self):
        embs = {}
        with open("data/DBP15K/glove.6B.300d.txt") as f:
            for line in tqdm.tqdm(f.readlines()):
                line = line.strip().split()
                embs[line[0]] = torch.tensor([float(x) for x in line[1:]])
        x1_path = os.path.join(self.root, self.pair, 'ent_ids_1')
        x2_path = os.path.join(self.root, self.pair, 'ent_ids_2')
        g1_path = os.path.join(self.root, self.pair, 'triples_1')
        g2_path = os.path.join(self.root, self.pair, 'triples_2')
        names_1 = os.path.join(self.root, self.pair,'ent_names_1')
        names_2 = os.path.join(self.root, self.pair,'ent_names_2')
        x1, edge_index1, rel1, assoc1 = self.process_graph(g1_path, x1_path, names_1,embs)
        x2, edge_index2, rel2, assoc2= self.process_graph(g2_path, x2_path, names_2,embs)
        pair_path = os.path.join(self.root, self.pair, 'ref_ent_ids')
        pair_set = self.process_pair(pair_path, assoc1, assoc2)
        pair_set = pair_set[:, torch.randperm(pair_set.size(1))]
        train_set = pair_set[:, :int(self.rate*pair_set.size(1))]
        # test_set = pair_set[:, int(self.rate2*pair_set.size(1)):int(0.1*pair_set.size(1))]
        test_set = pair_set[:, int(self.rate2*pair_set.size(1)):]
        val_set = pair_set[:, int(self.rate*pair_set.size(1)):int(self.rate2*pair_set.size(1))]
        if self.KG_num == 1:
            data = Data(x1=x1, edge_index1=edge_index1, rel1=rel1,assoc1=assoc1,
                        x2=x2, edge_index2=edge_index2, rel2=rel2,
                        train_set=train_set.t(), test_set=test_set.t(), val_set=val_set.t(), pair_set=pair_set.t())
        else:
            x = torch.cat([x1, x2], dim=0)
            edge_index = torch.cat([edge_index1, edge_index2+x1.size(0)], dim=1)
            rel = torch.cat([rel1, rel2+rel1.max()+1], dim=0)
            data = Data(x=x, edge_index=edge_index, rel=rel,train_set=train_set.t(), val_set=val_set.t(), test_set=test_set.t())
        torch.save(self.collate([data]), self.processed_paths[0])


    def loadNe(self,path):
        f1 = open(path)
        vectors = []
        for i, line in enumerate(f1):
            vect = line.rstrip()
            vect = np.fromstring(vect, sep=' ')
            vectors.append(vect)
        embeddings = np.vstack(vectors)
        embeddings=torch.tensor(embeddings)
        embeddings=torch.as_tensor(embeddings, dtype=torch.float32)
        return embeddings

    def process_graph(self, triple_path, ent_path, names,embs):
        g = read_txt_array(triple_path, sep='\t', dtype=torch.long)
        subj, rel, obj = g.t()
        assoc = torch.full((rel.max().item()+1,), -1, dtype=torch.long)
        assoc[rel.unique()] = torch.arange(rel.unique().size(0))
        rel = assoc[rel]
        idx = []
        with open(ent_path, 'r') as f:
            for line in f:
                info = line.strip().split('\t')
                idx.append(int(info[0]))
        idx = torch.tensor(idx)
        with open(ent_path, 'r') as f:
            ents = [int(line.strip().split('\t')[0]) for line in f.readlines()]
        ent_ref = {ent:i for i, ent in enumerate(ents)}
        x = [None for i in range(len(ents))]
        with open(names, 'r') as f:
            for line in f.readlines():
                try:
                    ent, name = line.strip().split('\t')
                except:
                    ent = line.strip()
                    name = ''
                ent_x = []
                for word in name.split():
                    word = word.lower()
                    if word in embs.keys():
                        ent_x.append(embs[word])
                if len(ent_x) > 0:
                    # x[ent_ref[int(ent)]] = torch.stack(ent_x, dim=0).mean(dim=0)+torch.rand(300)-0.5
                    # x[ent_ref[int(ent)]] = torch.stack(ent_x, dim=0).mean(dim=0)+torch.rand(300)*0.3333
                    x[ent_ref[int(ent)]] = torch.stack(ent_x, dim=0).mean(dim=0)
                else:
                    x[ent_ref[int(ent)]] = torch.rand(300)-0.5
        x = torch.stack(x, dim=0).contiguous()
        input_noises = Variable(torch.zeros(x.shape))
        input_noises.data.normal_(0, std=0.3)
        x = x + input_noises
        assoc = torch.full((idx.max().item()+1, ), -1, dtype=torch.long)
        assoc[idx] = torch.arange(idx.size(0))
        subj, obj = assoc[subj], assoc[obj]
        edge_index = torch.stack([subj, obj], dim=0)
        edge_index, rel = sort_edge_index(edge_index, rel)
        x = F.normalize(x, dim=1, p=2)
        return x, edge_index, rel, assoc
    def process_pair(self, path, assoc1, assoc2):
        e1, e2 = read_txt_array(path, sep='\t', dtype=torch.long).t()
        return torch.stack([assoc1[e1], assoc2[e2]], dim=0)
