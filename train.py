import os
import argparse
import itertools

import apex
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
from model import DSEA
from data import DBP15K
from loss import L1_Loss
from utils import add_inverse_rels, get_train_batch, get_hits1,get_hits2,get_hits_stable
from gan import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--data", default="data/DBP15K")
    parser.add_argument("--lang", default="zh_en")
    parser.add_argument("--rate", type=float, default=0.25)
    parser.add_argument("--r_hidden", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=3)
    parser.add_argument("--epoch", type=int, default=160)
    parser.add_argument("--neg_epoch", type=int, default=5)
    parser.add_argument("--test_epoch", type=int, default=5)
    parser.add_argument("--stable_test", action="store_true", default=False)
    args = parser.parse_args()
    return args

def init_data(args, device):
    data = DBP15K(args.data, args.lang, rate=args.rate)[0] 
    data.x1 = F.normalize(data.x1, dim=1, p=2).to(device).requires_grad_()
    data.x2 = F.normalize(data.x2, dim=1, p=2).to(device).requires_grad_()
    data.edge_index_all1, data.rel_all1 = add_inverse_rels(data.edge_index1, data.rel1)
    data.edge_index_all2, data.rel_all2 = add_inverse_rels(data.edge_index2, data.rel2)
    return data

def get_emb(model, data):
    model.eval()
    with torch.no_grad():
        x1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.rel_all1)
        x2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel_all2)
    return x1, x2
    
def train(model, criterion, optimizer, data,data_batch,train_batch):
    model.train()
    x1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.rel_all1)
    x2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel_all2)
    optimizer.zero_grad()
    loss = criterion(x1, x2, data_batch,train_batch)
    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()
    return loss
 
def test1 (model, data, stable=False):
    torch.cuda.empty_cache()
    with torch.no_grad():    
        x1, x2 = get_emb(model, data)
        print('-'*16+'Train_set'+'-'*16)
        get_hits1(x1, x2, data.train_set)
        torch.cuda.empty_cache()
        with torch.no_grad(): 
            print('-'*16+'Val_set'+'-'*17)
            get_hits1(x1, x2, data.val_set)
        print('-'*16+'Test_set'+'-'*17)
        torch.cuda.empty_cache()
        with torch.no_grad(): 
            get_hits1(x1, x2, data.test_set)

def construct_subgraph(a,b):
    graph = b
    for i in b[:,0]:
        index = torch.where(a[:,0]==i)[0]
        subgraph = a[index]
        graph = torch.cat([graph,subgraph],dim=0)
    return graph
         
def test(model, data, stable=False):
    torch.cuda.empty_cache()
    with torch.no_grad():    
        x1, x2 = get_emb(model, data)
        test_input_number,test_input,trust_input,negative_input1,negative_input2=get_hits2(x1, x2, data.test_set,data.train_set)
    return test_input_number,test_input,trust_input,negative_input1,negative_input2
            
def main(args):
    gc.collect()
    torch.cuda.empty_cache() 
    torch.autograd.set_detect_anomaly(True) 
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    data = init_data(args, device).to(device)
    batchsize = 64
    for epoch in range(args.epoch):
        torch.cuda.empty_cache()
        data.train_set=data.train_set.cuda()
        num_ite = len(data.train_set)//batchsize
        model =DSEA(data.x1.size(1), args.r_hidden).to(device)
        print(data.train_set.shape)
        optimizer = torch.optim.Adam(itertools.chain(model.parameters(), iter([data.x1, data.x2])))
        model, optimizer = apex.amp.initialize(model, optimizer)
        criterion = L1_Loss(args.gamma)
        x1, x2 = get_emb(model, data)
        loss_1=0
        for ite in range(num_ite):
            train_batch = get_train_batch(x1, x2,data.train_set[ite*batchsize:(ite+1)*batchsize], args.k)
            loss=train(model, criterion, optimizer, data,data.train_set[ite*batchsize:(ite+1)*batchsize], train_batch)
            loss_1 = float(loss)+loss_1
        loss_1=loss_1/num_ite
        print('Epoch:', epoch+1, '/', args.epoch, '\tLoss: %.3f'%loss_1, '\r', end='')
        test1(model, data,args.stable_test)
        print()      
        test_input_number,test_input,trust_input,negative_input1,negative_input2=test(model, data,args.stable_test)
        d = D(1).apply(weights_init).to(device)  
        g = G(1).apply(weights_init).to(device)  
        criterion_gan = nn.CrossEntropyLoss()
        d_optimizer = torch.optim.Adam(d.parameters(), lr=0.0004)
        g_optimizer = torch.optim.Adam(g.parameters(), lr=0.0001)
        gan_train(d, g, criterion_gan, d_optimizer, g_optimizer,trust_input,negative_input1,negative_input2,args.lang,epochs=100)
        if (epoch+1) % 3 == 0:
            print('--------------new data------------------')
            with torch.no_grad():
                d.load_state_dict(torch.load('data/DBP15K/'+args.lang+'/gan/d_90'))        
                trust_test = d(test_input).clone().detach()
                trust_test = torch.tensor(trust_test[:,0]>trust_test[:,1])+0
                new_alignment=test_input_number[torch.where(trust_test==1)[0]]
                # new_alignment=test_input_number[:5000,:]
                if len(new_alignment) < 1:
                    new_alignment=test_input_number[torch.where(trust_test==0)[0]][:500,:]
                    # new_alignment=construct_subgraph(data.pair_set,new_alignment)  
                if len(new_alignment) > 64:
                    # data.train_set=torch.cat([data.pair_set[17000:,:],new_alignment],dim=0)
                    new_alignment=construct_subgraph(data.train_set,new_alignment)  
                    data.train_set=new_alignment
                        
if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    main(args)
