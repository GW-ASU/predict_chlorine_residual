import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import itertools
import warnings
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import time
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
pd.options.mode.chained_assignment = None 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def norm(df1,df2, window_size):
    df2_con=pd.DataFrame(df2.reshape(-1, 5))
    df1_con=pd.DataFrame(df1.reshape(-1, 5))
    df1_norm=(df1_con-min(df2_con))/(max(df2_con)-min(df2_con))
    df1_norm_3d=np.array([df1_norm[i:i+window_size] for i in range (0,df1_norm.shape[0],window_size)])
    return df1_norm_3d

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.lstm = nn.LSTM(input_size = 5,
                              hidden_size = 512, 
                              num_layers =1, 
                              batch_first = True) 
       
        self.fc1 = nn.Linear(512, 60)
        self.dropout = nn.Dropout(p=0.3)
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        out, _ = self.lstm(x)
        out = self.dropout(out)  # Apply dropout after LSTM
        out = self.fc1(out[:, -1, :])  # Get the last output of the LSTM sequence
        return out
    
"'EARLY STOPPER'"
class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
"'LOAD LSTM'"
def istb4(device,para_dict):# if bidiretional true 2, if false 1
    mv_net = Model1()
    mv_net = mv_net.to(torch.float)
    mv_net.to(device)
    # criterion=torch.nn.SmoothL1Loss()
    # criterion=torch.nn.L1Loss()
    criterion=torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mv_net.parameters(), lr=para_dict['learning_rate'])
    return(mv_net,criterion,optimizer)

"' train model'"
def train_validation_lstm(para_dict,device,tensor_x_tr,tensor_y_tr,tensor_x_cv,tensor_y_cv):
    tr_mse=[]
    cv_mse=[]
    early_stopper = EarlyStopper(patience=100, min_delta=0)
    mv_net,criterion,optimizer= istb4(device,para_dict)
    for t in range(para_dict['train_episodes']):
        running_loss = []
        for b in range(0,len(tensor_x_tr),para_dict['batch_size']):
            inpt= tensor_x_tr[b:b+para_dict['batch_size'],:,:]
            target_tr= tensor_y_tr[b:b+para_dict['batch_size']]
            output_tr= mv_net(inpt)
            output_tr_1=output_tr.reshape(output_tr.shape[0],output_tr.shape[1])
            loss_tr = criterion(output_tr_1.to(device),target_tr)
            running_loss.append( loss_tr.item())           
            loss_tr.backward()
            optimizer.step()        
            optimizer.zero_grad() 

        output_cv= mv_net(tensor_x_cv) 
        output_cv_1=output_cv.reshape(output_cv.shape[0],output_cv.shape[1])

        loss_cv = criterion(output_cv_1.to(device),  tensor_y_cv)
        tr_mse.append(np.mean(running_loss))
        cv_mse.append(np.mean(loss_cv.item()))
        if early_stopper.early_stop(loss_cv.item()):             
            break
    return(tr_mse,cv_mse,mv_net)
