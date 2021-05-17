import torch
# torch.set_default_tensor_type(torch.cpu.FloatTensor)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
from dataloader import hogpretreatment

from dataloader import query_surpportSplit

from FFSTP_net import FFSTP
import torch
from dataloader import hogpretreatment

from dataloader import query_surpportSplit


def indexmask(label, data_index):
    import random
    index_ = [x[1] for x in data_index]
    return np.array([x[0] for x in data_index][random.choice([i for i, x in enumerate(index_) if x is label])])


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = ((a - b) ** 2).sum(dim=2)
    return -logits

def extract_data(path, n_class, n_support, n_query):
    SupportSet, QuerySet = query_surpportSplit(n_class, n_support, n_query, path)

    Support_loader = DataLoader(SupportSet, batch_size=300, shuffle=False)
    Query_loader = DataLoader(QuerySet, batch_size=100)

    return Support_loader, Query_loader
from torchsummary import summary

model = FFSTP().double()
model = nn.DataParallel(model)
resultdata = []
model.to(device)

optimizer = optim.Adamax(model.parameters(), lr=0.0001)
def train(epoch, model, Support_loader, Query_loader):
    global data_index
    model.train()
    for batch_idx, (data, target) in enumerate(Query_loader):
        data, target = data.to(device), target.to(device)
        hoglist = hogpretreatment(np.array(data.cpu().clone().numpy()))
        optimizer.zero_grad()
        hog_data = torch.tensor(np.array(hoglist),requires_grad=True).to(device)


        output =model(torch.unsqueeze(data, 1).double(),torch.unsqueeze(hog_data.clone(), 1)).to(device)
        output_supports = torch.zeros(0, requires_grad=True).double().to(device)
        target_supports = torch.zeros(0)
        for batch_idx, (data_support, target_support) in enumerate(Support_loader):
            data_support, target_support = data_support.to(device), target_support.to(device)
            hoglist_support = hogpretreatment(np.array(data_support.cpu().clone().numpy()))
            hog_data_support=torch.tensor(np.array(hoglist_support),requires_grad=True).to(device)
            output_support =model(torch.unsqueeze(data_support, 1).double(),torch.unsqueeze(hog_data_support.clone(), 1)).to(device)
            output_supports = torch.cat((output_supports.to(device), output_support.to(device)), 0)
            target_supports = torch.cat((target_supports.to(device), target_support.to(device)), 0)

        import random

        target_final = torch.zeros(0, requires_grad=True).to(device)
        chunk_counter = 0
        for x in torch.unique(target_supports, sorted=True):
            len_target_supports = len((target_supports == x).nonzero()[:, 0])
            target_final = torch.cat((target_final, torch.mean(output_supports[(target_supports == x).nonzero()[:, 0][
                torch.randperm(len_target_supports)[:int(torch.randint(1, len_target_supports, (1,))[0] + 1)]]],
                                                               0).float()), 0).to(device)

            chunk_counter += 1

        q = euclidean_metric(output, target_final.clone().view(chunk_counter, -1))
            #print(q)
        ##onehot = torch.nn.functional.one_hot(torch.tensor([x[0] for x in target]).long(), num_classes=5)

        loss = F.cross_entropy(q, torch.tensor([x[0] for x in target]).long().to(device))


        resultdata.append([epoch, loss])

        loss.backward()

        optimizer.step()
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(Query_loader.dataset),  # shotVec=[]
                       100. * batch_idx / len(Query_loader), loss.item()))  # hog_out = model(hog_data)
        elif batch_idx % 1000 == 0:
            PATH = str(batch_idx) + '.pth'
            torch.save(model.state_dict(), PATH)

def test(epoch, model, Support_loader, Query_loader):
    #global data_index
    #model.train()
    for batch_idx, (data, target) in enumerate(Query_loader):
        data, target = data.to(device), target.to(device)
        hoglist = hogpretreatment(np.array(data.cpu().clone().numpy()))
        optimizer.zero_grad()
        hog_data = torch.tensor(np.array(hoglist),requires_grad=True).to(device)


        output =model(torch.unsqueeze(data, 1).double(),torch.unsqueeze(hog_data.clone(), 1)).to(device)
        output_supports = torch.zeros(0, requires_grad=True).double().to(device)
        target_supports = torch.zeros(0)
        for batch_idx, (data_support, target_support) in enumerate(Support_loader):
            data_support, target_support = data_support.to(device), target_support.to(device)
            hoglist_support = hogpretreatment(np.array(data_support.cpu().clone().numpy()))
            hog_data_support=torch.tensor(np.array(hoglist_support),requires_grad=True).to(device)
            output_support =model(torch.unsqueeze(data_support, 1).double(),torch.unsqueeze(hog_data_support.clone(), 1)).to(device)
            output_supports = torch.cat((output_supports.to(device), output_support.to(device)), 0)
            target_supports = torch.cat((target_supports.to(device), target_support.to(device)), 0)

        import random

        target_final = torch.zeros(0, requires_grad=True).to(device)
        chunk_counter = 0
        for x in torch.unique(target_supports, sorted=True):
            len_target_supports = len((target_supports == x).nonzero()[:, 0])
            target_final = torch.cat((target_final, torch.mean(output_supports[(target_supports == x).nonzero()[:, 0][
                torch.randperm(len_target_supports)[:int(torch.randint(1, len_target_supports, (1,))[0] + 1)]]],
                                                               0).float()), 0).to(device)

            chunk_counter += 1

        q = euclidean_metric(output, target_final.clone().view(chunk_counter, -1))
            #print(q)
        ##onehot = torch.nn.functional.one_hot(torch.tensor([x[0] for x in target]).long(), num_classes=5)

        loss = F.cross_entropy(q, torch.tensor([x[0] for x in target]).long().to(device))
        print('test:')
        print(loss)
        return loss.numpy()


loss = float('inf')
ini_SupportSet, ini_QuerySet = extract_data('sample_training_data', 2, 5, 5)
SupportSet, QuerySet = ini_SupportSet, ini_QuerySet
testlist=[]
for epoch in range(1, 1000+ 1)
    if epoch % 5 == 0:#step include before shuffle
        SupportSet, QuerySet =extract_data('sample_training_data', 2, 5, 5)
            #test(epoch, model, TestSupportSet, TestQuerySet )
    if epoch % 100 == 0:
       
        PATH = str(epoch) + '.pth'
        with torch.no_grad():
            TestSupportSet, TestQuerySet = extract_data('sample_test_data', 2, 2, 1)
            testlist.append([epoch, test(epoch, model, TestSupportSet, TestQuerySet)])
        torch.save(model.state_dict(), PATH)
        torch.cuda.empty_cache()
        import pandas as pd

        df = pd.DataFrame(resultdata, columns=['epoch', 'loss'])
        df.to_csv("resultdata1.csv")
        import pandas as pd

        df = pd.DataFrame(testlist, columns=['epoch', 'loss'])
        df.to_csv("Testdata1.csv")

    train(epoch, model, SupportSet, QuerySet)

