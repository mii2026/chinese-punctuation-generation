import torch
import os
from data import preprocessing
from torch.utils.data import DataLoader
from torch import nn, optim
from model import CNNLstmBert
from tqdm import tqdm
import numpy as np
from sklearn import metrics 
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from scl_loss import ce_scl_loss
from focal_loss import FocalLoss
import json

# random seeds
seed = 20
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic=True

# set parameters
with open('config.json') as f:
    config = json.load(f)

sequence_len = config['sequence_len']
batch_size = config['batch_size']
output_size = 15
epochs = config['epochs']
learning_rate = config['learning_rate']
loss_function = config['loss_function']

model_save_path = config['model_save_path']
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

train_data_path = config['train_data_path']
valid_data_path = config['valid_data_path']

# define punc label
puncs = [i for i in range(15)]

# Prepare dataloader
print('Preprocess data')

train_datasets = []
for i in train_data_path:
    with open(i, 'r', encoding='utf-8') as f:
        train_data = f.readlines()
    train_datasets.append(train_data)

valid_datasets = []
for i in valid_data_path:
    with open(i, 'r', encoding='utf-8') as f:
        valid_data = f.readlines()
    valid_datasets.append(valid_data)

trainset = preprocessing(train_datasets, sequence_len)
validset = preprocessing(valid_datasets, sequence_len)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=None)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=None)

# prepare model
print('Load pretrained model')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Current device: {device}')
model = CNNLstmBert(output_size).to(device)
for p in model.bert1.parameters():
    p.requires_grad = True

# prepare optimizer and criterion
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
if loss_function == 'focal':
    criterion = FocalLoss(gamma=2)
    print('focal')
else:
    criterion = nn.CrossEntropyLoss()
    print('cross_entropy')

# make log file
with open(model_save_path+'/log.csv', 'w') as f:
    f.write('epoch\ttraining_loss\tval_loss\tf1_socres\n')

# training with valid
print('Start Training')
best_f1 = 0
for e in range(epochs):
    # training
    model.train()
    train_loss = []
    for inputs, labels in tqdm(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        h0, output = model(inputs, device)
        if loss_function == 'scl':
            labels = labels.view(-1)
            loss = ce_scl_loss(output.reshape(-1, 15),
                            labels.cuda().reshape(-1), h0,
                            lambda_value=0.1,
                            temperature=0.3,
                            pooling=False,
                            weight=None,
                            device=device)
            train_loss.append(loss.cpu().data.numpy())
            loss.mean().backward()
        else:
            labels = labels.view(-1)
            loss = criterion(output, labels)
            train_loss.append(loss.cpu().data.numpy())
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    #evaluate validate
    val_loss = []
    val_acc = []
    val_f1 = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(validloader):
            inputs, labels = inputs.to(device), labels.to(device)
            _, output = model(inputs, device)
            labels = labels.view(-1)
            loss = criterion(output, labels)
            val_loss.append(loss.cpu().data.numpy())

            y_preds = output.argmax(dim=1).cpu().data.numpy().flatten()
            y_labels = labels.cpu().data.numpy().flatten()
            val_f1.append(metrics.f1_score(y_labels, y_preds, average=None, labels=puncs))

    # save loss
    train_loss = np.mean(train_loss)
    val_loss = np.mean(val_loss)
    val_f1 = np.array(val_f1).mean(axis=0)
    with open(model_save_path+'/log.csv', 'a') as f:
        f1 = '/'.join(['{:4f}'.format(i) for i in val_f1])
        f.write('{}\t\t{:4f}\t{:4f}\t{}\n'.format(e+1, train_loss, val_loss, f1))

    #save model
    torch.save(model.state_dict(), model_save_path+f"/model_{e}epoch")

    #save best model
    if np.mean(val_f1[1:]) > best_f1:
        torch.save(model.state_dict(), model_save_path+f"/model_best")
        best_f1 = np.mean(val_f1[1:])