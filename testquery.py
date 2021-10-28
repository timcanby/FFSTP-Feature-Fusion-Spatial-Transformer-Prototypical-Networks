# coding: utf-8
import os
import sys
import random
import string

this_dir = os.path.abspath(os.path.dirname(__file__))

#model_file = os.path.join(this_dir, 'trained_model/500/5390.pth')
import pandas as pd
import torch.nn as nn

import numpy as np
import torch.optim as optim

from FFSTP_net import FFSTP

from sklearn.ensemble import VotingClassifier

from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from dataloader import imputOne_data
dt = datetime.now()
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
device = "cpu"
import torch
from dataloader import hogpretreatment
import os

this_dir = os.path.abspath(os.path.dirname(__file__))

model_file = os.path.join(this_dir, 'trained_model/1760.pth')
from shutil import copyfile
def load_models():


    model = FFSTP().double()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    if not os.path.exists(model_file):
        raise FileNotFoundError(f'モデルファイルが見つからなかった.モデルディレクトリ：{os.path.abspath(model_file)}')

    return model

model=load_models()
X = np.load(os.path.join(this_dir, 'models/X.npy'))
y = np.load(os.path.join(this_dir, 'models/y.npy'))
labellist =np.load(os.path.join(this_dir, 'models/label.npy'))
estimators = [
    ('logit', LogisticRegression(solver='lbfgs', max_iter=10)),
    ('knn', KNeighborsClassifier()), ]
optimizer = optim.Adam(model.parameters(), lr=0.001)


def testour(path,groundtruth):

    querySet, testlabel = imputOne_data(path)
    Query_loader = DataLoader(querySet)
    global data_index
    model.train()
    output=[]


    for batch_idx, (data, target) in enumerate(Query_loader):
        hoglist = hogpretreatment(np.array(data))
        data, target = torch.from_numpy(np.array(data)).cpu(), torch.tensor(target).cpu()
        hog_data = torch.from_numpy(np.array(hoglist)).clone().cpu()


        optimizer.zero_grad()
        output.append(model(torch.from_numpy(np.expand_dims(data, 1)).double().clone(),
                       torch.from_numpy(np.expand_dims(hog_data, 1)).double().clone()).cpu())

    voting = VotingClassifier(estimators, voting='soft')
    voting.fit(X, y)

    y_pred_s = voting.predict_proba(output[-1].detach().numpy())
    order = np.argsort(y_pred_s, axis=1)

    r=  list(reversed([each for each in labellist[:, 1][order[:, -20:]][0]]))
    r1=  list(reversed([each for each in labellist[:, 1][order[:,:]][0]]))
    print(r)

    counter1=1
    resut=0
    for each1 in r1 :

        if groundtruth in each1:
            print(counter1)
            resut=counter1
        counter1+=1


    text = r[0].split('_')[0]
    options = [{'text': i.split('_')[0]+':', 'confidence': i.split('_')[0]} for i in r]
    print({'text': text, 'options': options})
    return resut

outputs= []
train_examples = []
imagelabel = []
labels = []
mrr = 0
w = 0
out = []
def getNamelist(dataPath):
    csvreader = pd.read_csv(dataPath, header=None)
    final_list = csvreader.values.tolist()
    return final_list
for eachData in getNamelist('query_rank.csv')[1:]:

        mrr += 1

        w += 1 / (testour(eachData[0],eachData[1]) + 1)
        print(eachData[0])
        print(eachData[1])
        outputs.append([eachData[0], eachData[1], testour(eachData[0],eachData[1]) + 1])




df=pd.DataFrame(outputs,columns=['groundTruth','result','index'])
df.to_csv("our_mrr.csv")
print('our_mrr')

print(w / (mrr))