import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
import numpy as np


def add_inverse_rels(edge_index, rel):
    edge_index_all = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    rel_all = torch.cat([rel, rel+rel.max()+1])
    return edge_index_all, rel_all

def get_train_batch(x1, x2,train_set, k=6):
    e1_neg1 = torch.cdist(x1[train_set[:, 0]], x1, p=1).topk(k+1, largest=False)[1].t()[1:]
    e1_neg2 = torch.cdist(x1[train_set[:, 0]], x2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg1 = torch.cdist(x2[train_set[:, 1]], x2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg2 = torch.cdist(x2[train_set[:, 1]], x1, p=1).topk(k+1, largest=False)[1].t()[1:]  
    train_batch = torch.stack([e1_neg1, e1_neg2, e2_neg1, e2_neg2,], dim=0)
    return train_batch

def softmax(x, axis):
    x = x - torch.max(x,axis)[0].unsqueeze(1)
    y = torch.exp(x)
    return y / torch.sum(y,axis).unsqueeze(1)

def concat(a,b,c):
    d=torch.cat((a.unsqueeze(0),b.unsqueeze(0),c.unsqueeze(0)),0)
    return d

def dis(x, y):
    return torch.sum(torch.abs(x-y), dim=-1)

def get_hits1(x1, x2, pair, dist='L1', Hn_nums=(1, 5)):
    pair_num = pair.size(0)
    # S = torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1).cpu()
    S = torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1)
    for k in Hn_nums:
        pred_topk= S.topk(k, largest=False)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
        print('Hits@%d: %.2f%%    ' % (k, Hk*100),end='')
    rank = torch.where(S.sort()[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = (1/(rank+1)).mean().item()
    print('MRR: %.3f' % MRR)  
    
def get_hits2(x1, x2, pair,trainset, dist='L1', Hn_nums=(1, 10)):
    S = torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1)
    for k in Hn_nums:
        if k == 1:
            test_input = S.topk(k, largest=False)[0]
            number=S.topk(k, largest=False)[1]
            test_input_number=torch.cat([pair[:, 0].unsqueeze(1),pair[number].squeeze()[:,1].unsqueeze(1)],dim=1)
            test_value = torch.topk(test_input.squeeze(), 6000, largest=False, sorted=False)
            test_input = test_value[0].unsqueeze(1)
            test_input_number = test_input_number[test_value[1]]
            trust_input=dis(x1[trainset[:, 0]],x2[trainset[:, 1]])
            # print(torch.cdist(x1[trainset[:, 0]], x2[trainset[:, 1]], p=1).topk(2, largest=False)[0])
            # negative_input1=torch.cdist(x1[trainset[:, 0]], x2[trainset[:, 1]], p=1).topk(2, largest=False)[1][:,1].squeeze()
            # negative_input1=dis(x1[trainset[:, 0]], x2[trainset[negative_input1][:,1]])
            # negative_input2=torch.cdist(x2[trainset[:, 1]], x1[trainset[:, 0]], p=1).topk(2, largest=False)[1][:,1].squeeze()
            # negative_input2=dis(x2[trainset[:, 1]], x1[trainset[negative_input2][:,0]])
            # negative_input1=torch.cdist(x1[trainset[:, 0]], x2[trainset[:, 1]], p=1).topk(2, largest=False)[0][:,1].squeeze()
            negative_input1=torch.cdist(x1[pair[:, 0]], x2[trainset[:, 1]], p=1).topk(2, largest=False)[0][:,1].squeeze()
            negative_input2=torch.cdist(x2[trainset[:, 1]], x1[trainset[:, 0]], p=1).topk(2, largest=False)[0][:,1].squeeze()
    #     Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
    #     print('Hits@%d: %.2f%%    ' % (k, Hk*100),end='')
    # rank = torch.where(S.sort()[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    # MRR = (1/(rank+1)).mean().item()
    # print('MRR: %.3f' % MRR)
    return test_input_number,test_input,trust_input,negative_input1,negative_input2
  
def get_hits_stable(x1, x2,pair):
    pair_num = pair.size(0)
    S = F.normalize(torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1),dim=0) 
    index = (S.softmax(1)+S.softmax(0)).flatten().argsort(descending=True) 
    index_e1 = index//pair_num 
    index_e2 = index%pair_num 
    aligned_e1 = torch.zeros(pair_num, dtype=torch.bool)
    aligned_e2 = torch.zeros(pair_num, dtype=torch.bool)
    true_aligned = 0
    for _ in range(pair_num*100):
        if aligned_e1[index_e1[_]] or aligned_e2[index_e2[_]]:
            continue
        if index_e1[_] == index_e2[_]:
            true_aligned += 1
        aligned_e1[index_e1[_]] = True
        aligned_e2[index_e2[_]] = True
    print('Both:\tHits@Stable: %.2f%%   ' % (true_aligned/pair_num*100))
