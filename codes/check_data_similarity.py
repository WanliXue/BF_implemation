from BF_TS import BF_TS
import scipy.io
import numpy as np
from svmutil import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
import math
from scipy.spatial.distance import pdist
import scipy
import matplotlib as mpl
mpl.use('TkAgg')  #for mac using matplotlib
import matplotlib.pyplot as plt

# MNIST
# data=scipy.io.loadmat('../data/10mnist_data.mat')
# # print data['test_set'].shape #(1000,784)
# train_data=data['train_set']
# print train_data.shape  #[6000,784]
# test_data=data['test_set']
# train_label=data['train_label']
# test_label=data['test_label']


# HAR
data_train=np.load('../data/HAR_traindata.npy') #(7352,561)
# data_test=np.load('../data/HAR_testdata.npy') #(2947 , 561)


#CSI
data_train=np.load('../data/CSI_traindata.npy') #(150,10800)
# print data_train.shape

test=data_train[0:10,:]

# dists=[]
# for row in train_data[0:10,:]:
#     # print row.shape
#     dists.append()

dist = scipy.spatial.distance.cosine(data_train[0,:],data_train[1,:])
print 'first one ',dist

# data=np.append(data_train,data_test,axis=0)
# print 'data shape ',data.shape
data=data_train

dists=pdist(data,metric='cosine')
print dists
print dists.shape
print max(dists)

r1=( dists >= 0.95 )
# print r1
# print r1.sum()

r2= (dists <= 1)

r = np.logical_and(r1,r2)

print 'r is ', r
print 'r sum is ',r.sum()

#r1_true= [i for i, x in enumerate(r1) if x]

# dists2 = pdist(data_test, 'correlation')
# print dists2
# print dists2.shape
# print max(dists2)
# r2=(dists2 >= 0.8)
# print r2.sum()

# r2_true= [i for i, x in enumerate(r2) if x]

# # intersection = list(set(r1_true) & set(r2_true))
# intersection=set(r1_true).intersection(r2_true)
# print len(intersection)

# if set(intersection)==set(r1_true):
#     print True
# else:
#     print False

z=scipy.spatial.distance.squareform(r)
print z.shape
print z

upper_tria = np.triu_indices(150)

z[upper_tria]= False


equal_pairs=np.argwhere(z==True)
print equal_pairs.shape
print equal_pairs[0], equal_pairs[1]


a=[x[0] for x in equal_pairs]
print a

b=[x[1] for x in equal_pairs]
print b
print len(b)

unique_a=set(a)
# print unique_a
print len(unique_a)

unique_b=set(b)
# print unique_b
print len(unique_b)

# unique=set(a) | set(b)
unique=set(a).union(set(b))
# print unique
print len(unique)
#
