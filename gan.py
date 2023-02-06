import os
import PIL
import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.autograd import Variable

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.01)
    # elif classname.find('Softmax') != -1:
    #     nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     nn.init.constant_(m.bias.data, 0)

class D(nn.Module):
    def __init__(self, input_size, output_size=2):
        super(D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.BatchNorm1d(input_size*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(input_size * 2, input_size * 4),
            nn.BatchNorm1d(input_size*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(input_size * 4, input_size * 8),
            nn.BatchNorm1d(input_size*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(input_size * 8, output_size),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.as_tensor(self.layer4(out),dtype=torch.float32)
        return out

class G(nn.Module):
    def __init__(self, input_size):
        super(G, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size*2),
            nn.BatchNorm1d(input_size*2),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Linear(input_size * 2, input_size * 4),
            nn.BatchNorm1d(input_size*4),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(input_size * 4, input_size * 8),
            nn.BatchNorm1d(input_size*8),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(input_size * 8, input_size * 16),
            nn.BatchNorm1d(input_size*16),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(input_size * 16, input_size),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.linear(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

def gan_train(d, g, criterion, d_optimizer, g_optimizer,trust_input,negative_input1,negative_input2, KG,epochs,print_every=10):
    device = 'cuda'
    iter_count = 0
    negative_inputs = torch.cat([negative_input1,negative_input2],dim=0)
    trust_input = trust_input.unsqueeze(1)
    negative_inputs = negative_inputs.unsqueeze(1)
    negative_inputs = torch.cat([trust_input,negative_inputs],dim=0)
    for epoch in range(epochs):
        real_inputs = trust_input.to(device)         
        input_noises = Variable(torch.zeros(real_inputs.shape).cuda())
        input_noises.data.normal_(0, std=0.5)
        real_inputs = real_inputs + input_noises
        fake_noises = Variable(torch.zeros(negative_inputs.shape).cuda())
        fake_noises.data.normal_(0, std=0.5)
        negative_inputs = negative_inputs + fake_noises
        fake_inputs = g(negative_inputs.to(device))  
        real_labels = torch.ones(real_inputs.size(0),dtype=torch.long).to(device)           
        fake_labels = torch.zeros(fake_inputs.size(0),dtype=torch.long).to(device)          

        d_output_real = d(real_inputs)                         
        d_output_real = torch.as_tensor(d_output_real,dtype=torch.float32)
        d_loss_real = criterion(d_output_real, real_labels)              
        d_output_fake = d(fake_inputs.detach())                  
        d_output_fake = torch.as_tensor(d_output_fake,dtype=torch.float32)
        d_loss_fake = criterion(d_output_fake, fake_labels)                
        d_loss = 0.5 * d_loss_fake + 0.5 * d_loss_real                                
        d_optimizer.zero_grad()                                           
        d_loss.backward()                                                  
        d_optimizer.step()                                              

        real_labels = torch.ones(real_inputs.size(0),dtype=torch.long).to(device) 
        real_inputs = real_inputs.to(device) 
        real_labels = torch.ones(real_inputs.size(0),dtype=torch.long).to(device) 
        fake_inputs = g(real_inputs)  
        g_output_fake = d(fake_inputs)                          
        g_output_fake = torch.as_tensor(g_output_fake,dtype=torch.float32)
        g_loss = criterion(g_output_fake, real_labels)                     
        g_optimizer.zero_grad()                                            
        g_loss.backward()                                                
        g_optimizer.step()                                                 
        if iter_count % print_every == 0:
            torch.save(d.state_dict(), 'data/DBP15K/'+KG+'/gan/d_' + str(epoch))                         
        iter_count += 1
        
    


