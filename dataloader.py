import torch

from datetime import datetime
dt=datetime.now()
from Sampler import samplerset
from torch.utils.data import DataLoader
from skimage.transform import rescale, resize
#from preprocessing import thinning

import numpy as np
import torch.utils.data as utils
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import cv2
import numpy as np
from skimage.feature import hog
img_size=50#define the image size
from torchvision import datasets, transforms
from skimage.color import rgb2gray

def hogpretreatment(image):
    global img_size
    hoglist=[]
    for eachpic_inbatch in image:
        fd, hog_image = hog(np.array(eachpic_inbatch).reshape(img_size,img_size), orientations=8, pixels_per_cell=(6, 6),
                            cells_per_block=(2, 2), visualize=True, feature_vector=True)
        hoglist.append(fd)
    return hoglist
from PIL import Image
import scipy
from skimage import io
def loadimage(path):
    #thinning(path)

    global img_size

    camera = io.imread(path,0)

    out_im = np.array(resize(rgb2gray(camera),(50,50)), dtype='float64').reshape(-1, img_size, img_size)


    return out_im

import os
from sklearn.model_selection import train_test_split
def init_dataloader(n_class, n_sample,dic_path):

    labelset = []
    for root, dirs, files in os.walk(dic_path):
        for d in dirs:
            labelset.append(os.path.join(root, d))

    Rlist=samplerset(n_class, n_sample, labelset)
    datalist=[]
    labellist=[]

    for class_idx, (label,data) in enumerate(Rlist):

        for eachpic in data:
            datalist.append([loadimage(eachpic),class_idx])
            labellist.append([class_idx,label])


    return  datalist,labellist
def init_dataloaderpca(n_class, n_sample,dic_path):

    labelset = []
    for root, dirs, files in os.walk(dic_path):
        for d in dirs:
            labelset.append(os.path.join(root, d))

    Rlist=samplerset(n_class, n_sample, labelset)
    datalist=[]
    labellist=[]

    for class_idx, (label,data) in enumerate(Rlist):

        for eachpic in data:
            datalist.append([loadimage(eachpic).flatten()])
            labellist.append([class_idx,label])


    return  np.squeeze(datalist),labellist
def transform2Tensor(data,label):
    tensor_x = torch.stack([torch.Tensor(i) for i in data])  # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor(i) for i in label])

    my_dataset = utils.TensorDataset(tensor_x, tensor_y)  # create your datset


    return my_dataset

#datalist=imagefeature,index labellist=index,image_path
def query_surpportSplit(n_class,n_query,n_support,dic_path):
    datalist,labellist=init_dataloader(n_class,(n_query+n_support),dic_path)
    a,b= np.hsplit(np.array(datalist),[-1])
    from sklearn.model_selection import train_test_split
    X_support, X_query, y_support, y_query = train_test_split(a, b, test_size=n_support*n_class, stratify=b)
    SupportSet=transform2Tensor(np.squeeze(np.array(X_support.tolist())),np.array(y_support.tolist()))
    QuerySet=transform2Tensor(np.squeeze(np.array(X_query.tolist())),np.array(y_query.tolist()))
    return SupportSet,QuerySet

def test_dataset(n_class,n_query,n_support,dic_path):
    datalist,labellist=init_dataloader(n_class,(n_query+n_support),dic_path)
    a,b= np.hsplit(np.array(datalist),[-1])

    SupportSet=transform2Tensor(np.squeeze(np.array(a.tolist())),np.array(b.tolist()))

    return SupportSet,labellist

