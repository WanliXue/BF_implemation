from l1 import l1
from cvxopt import normal
# import os
import time
import scipy.io
# import pandas
import numpy as np
import math
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from SRC_library import  SRC
from BF_TS import BF_TS

#-------SRC-------------
com_to_len=300

data=scipy.io.loadmat('../data/raw_face.mat')

data_train=data['training_temp']
data_test = data['trainingtest_temp']
label_train = data['class_label']
label_test = data['class_label_test']


label_train=label_train[0]
label_test=label_test[0]

# print data_train.shape  # (2016, 1216)
# print data_test.shape   # (2016, 380)
# print label_train.shape
# print label_test.shape

# -------- SVD the data set get the compression matrix---------
u,s,vh = np.linalg.svd(data_train,full_matrices= True)
# print u.shape

# get the compression matrix
compress_matrix = u[:,0:com_to_len].T
# print compress_matrix.shape  # (com_to_len , 2016)
#---------------

# compressed_data=np.array( data_train)
# compressed_test_data=np.array(data_test)

# print 'Compress'
compressed_data = np.matmul(compress_matrix,data_train)
# print compressed_data.shape  #(com_to_len,1216)
compressed_test_data=np.matmul(compress_matrix,data_test)
# print compressed_test_data.shape #(com_to_len,380 )
# print 'Compress done'


src= SRC()
# accuracy, _,_ = src.run_src_classifier(compressed_data,compressed_test_data,label_train,label_test)
accuracy, _,_ = src.run_src_classifier_bf(compressed_data,compressed_test_data,label_train,label_test)
print accuracy
# ------------



# #import numpy as np
# import numpy.linalg
# import scipy.io
# import unittest
# from pyCSalgos.BP.l1eq_pd import l1eq_pd
#
#
# data=scipy.io.loadmat('../data/csi.mat')
#
# data_train=np.load('../data/CSI_train_inputdata.npy')
# data_test = np.load('../data/CSI_test_inputdata.npy')
#
# label_train = data['train_label']
# label_test = data['test_label']
#
# # print data_train.shape
# # print data_train
#
#
#
# src=SRC()
# acc,_,_=src.run_src_classifier_bf(data_train.T,data_test.T,label_train,label_test )
#


# Y=data_test.T
# A=data_train.T
# # print Y.shape #(100000,50)
# # print A.shape
#
# mu,sigma =0,0.1
# ran_m=np.random.normal(mu,sigma,(100,10000))
# print ran_m.shape
#
# A=np.matmul(ran_m,A)
# Y=np.matmul(ran_m,Y)
# print Y.shape
# print A.shape
#
# # x0=A/y
# x0=np.matmul(A.T,Y)
# # print x0
# print x0.shape
#
#
#
#
#
# # xr = l1eq_pd(X0[:,i], A, np.array([]), Y[:,i])
# print Y[:,0]
# print Y[:,0].shape
#
#
#
# # xr = l1eq_pd(x0[:,0], A, np.array([]), Y[:,0])
# #
# # print xr.shape
# # print xr
# #
# # err1 = numpy.linalg.norm(Y[:,0] - np.dot(A,xr))
# # print 'err =',err1




# # --------- SRC attemp ------
# # for the total test samples number
# _,samples_num = compressed_test_data.shape
# # print samples_num
#
# # do the for loop and calcualte src for each one
#
#
# #  sovle the A*x = b one by one
# A = compressed_data
# # print A.shape
#
# class_num = len(np.unique(label_train))
#
# right = 0 # predict ccorrect
# wrong = 0 #~
#
#
#
#
# for test_ind in range(0,samples_num):
#
#     print 'start predict sample ',test_ind
#     # test_ind=2 #can try only one as example then comment for loop
#
#
#     b=compressed_test_data.T[test_ind]
#     # print b.shape
#
#     # A * coef = b
#     coef = cvx.Variable(1216)
#     obj = cvx.Minimize(cvx.norm(coef,1))
#     const = [A * coef == b]
#     prob = cvx.Problem(obj,const)
#     result = prob.solve()
#     # print coef.value.size
#     # print type(coef.value)
#     # print coef.value
#
#     # # ---temp---
#     # tempCoef = np.zeros(len(label_train))
#     #
#     # # tempCoef[index_of_class] = 1
#     # # print tempCoef[31], tempCoef[32]
#     #
#     # print label_train.shape
#     #
#     # # print index which label eaquals to what we want
#     # index_of_class = np.where(label_train == 1)[0]
#     # print index_of_class
#     #
#     #
#     # # put the coef's value into the tempCoef with correct index
#     # np.put(tempCoef,index_of_class,coef.value[index_of_class])
#     # # print tempCoef
#     #
#     # temp_b = np.matmul(A, tempCoef)
#     # print LA.norm(temp_b-b,2)
#     # # -----
#
#     # save the residual for total 38 class and find which one closer(smaller)
#     res=np.zeros(class_num)
#
#     for i in range(1,class_num+1):
#         tempCoef=np.zeros(len(label_train))
#         # find the index of that class
#         index_of_class = np.where(label_train == i)[0]
#         # put the coef's value into the tempCoef with correct index
#         np.put(tempCoef,index_of_class,coef.value[index_of_class])
#         temp_b=np.matmul(A,tempCoef)
#         res[i-1] = LA.norm(temp_b-b,2)
#
#
#     min_index = np.argmin(res)
#     # print min_index+1  #predicted label
#     # print label_test[test_ind]  # true label
#
#     # the predict class start from 0
#     if (min_index +1 == label_test[test_ind]): #predict true
#         right += 1
#
#     else:
#         wrong +=1
#         print 'wrong!'
#
# print 'total got: ', right+wrong
# print 'wrong is: ', wrong
# print 'accuracy is: ', float(right)/(right+wrong)
#
#
#
# #--------------------


# ----------------------------------
#
# import numpy
# import cvxpy as cvx
#
# m = 50
# n = 100
# numpy.random.seed(1)
# A = numpy.random.randn(m, n)
# b = numpy.random.randn(m)
#
# x = cvx.Variable(n)
# obj = cvx.Minimize(cvx.norm(x,1))
# const = [A * x == b]
# prob = cvx.Problem(obj,const)
# result = prob.solve()
#
# # print x.value
# print x.size
#



# ------------
from l1 import l1
# from cvxopt import normal
#
# m, n = 50, 100
# P, q = normal(m,n), normal(m,1)
# # print type(P)
# # # u = l1(P,q)