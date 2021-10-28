# coding: utf-8
import os
import sys
import random
import string
from vit_pytorch import ViT
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
X = np.load(os.path.join(this_dir, 'vgg_X1.npy'))
y = np.load(os.path.join(this_dir, 'vgg_y1.npy'))
labellist =np.load(os.path.join(this_dir, 'vgg_labels.npy'))
estimators = [
    ('logit', LogisticRegression(solver='lbfgs', max_iter=10)),
    ('knn', KNeighborsClassifier()), ]
optimizer = optim.Adam(model.parameters(), lr=0.001)
from PIL import Image
import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
def testour(path,groundtruth):
    outputs = []
    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    return_layers = {
        'mlp_head': 'lg_mlp_head',

    }
    mid_getter = MidGetter(v, return_layers=return_layers, keep_output=True)
    image = Image.open(path).resize((256, 256))
    x = TF.to_tensor(image)
    x.unsqueeze_(0)



    voting = VotingClassifier(estimators, voting='soft')

    voting.fit(X.squeeze(), y)
    mid_outputs, model_output = mid_getter(x)

    y_pred_s = voting.predict_proba(mid_outputs['lg_mlp_head'][-1].to('cpu').detach().numpy().copy().flatten().reshape(1, -1))
    order = np.argsort(y_pred_s)



    #r1=  list(reversed([each for each in labellist[:, 1][order[:,:]][0]]))
    #print(r1)

    counter1=1
    resut=0
    for each1 in labellist[order[0]] :
        #print(each1)

        if groundtruth in each1:
            print(counter1)
            resut=counter1
        counter1+=1


    #text = r[0].split('_')[0]
    #options = [{'text': i.split('_')[0]+':', 'confidence': i.split('_')[0]} for i in r]
    #print({'text': text, 'options': options})
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
        outputs.append([eachData[0], eachData[1], testour(eachData[0],eachData[1])+1])




df=pd.DataFrame(outputs,columns=['groundTruth','result','index'])
df.to_csv("vgg_mrr.csv")
print('vgg_mrr')

print(w / (mrr))