import random
import pandas as pd
import os



def getfileFromfilter(rootdir):
    import os
    list = os.listdir(rootdir)
    ReturnList=[]
    for i in range(0, len(list)):
        if list[i]!='.DS_Store':

            path = os.path.join(rootdir, list[i])
            ReturnList.append(path)
    return (ReturnList)

#Enter the names of all the folders to be sampled ：dic_path should be a set(list) of folder name(one folder for one class)
#If the number of samples exceeds the number of data present in the folder, all files in the folder are returned

def samplerset(n_class,n_sample,dic_path):
    classlist=random.sample(list(set(dic_path)), n_class)
    R_list=[]
    for eachclass in classlist:
        piclist=[]
        labeltxt=eachclass.split('/')[-1]
        for root, dirs, files in os.walk(eachclass):
            for f in files:
                piclist.append(os.path.join(root, f))
        if n_sample<=len(piclist):
            R_list.append([labeltxt, random.sample(list(set(piclist)), n_sample)])
        else:R_list.append([labeltxt, random.sample(list(set(piclist)),len(piclist))])
        saveSample_csv(R_list)#save samplelist
    return R_list


def saveSample_csv(R_list):
    df = pd.DataFrame(R_list, columns=["label",'image_path'])
    df.to_csv("sample_result.csv")



