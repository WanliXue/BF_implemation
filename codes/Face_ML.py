# import os
import time
import scipy.io
# import pandas
import numpy as np
import math
from svmutil import *
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from SRC_library import SRC

from BF_TS import BF_TS


data=scipy.io.loadmat('../data/face.mat')

data_train=data['training']
data_test = data['trainingtest']
label_train = data['class_label']
label_test = data['class_label_test']

# print data_train.shape  # (300, 1216)
# print data_test.shape   # (300, 380)
# print label_train.shape
# print label_test.shape

data_train = data_train.T
data_test=data_test.T


lengh_range=[8000, 10000, 20000, 50000, 80000]

# for i in range(len(lengh_range)):


# # ------ BF ---------
length = 30000
# length = lengh_range[i]
print 'iteration coming to ', 30000
b = 5
num_hash = 2
dis = float(5)

# generate the npy with the bf and data
bf_ts=BF_TS(length,num_hash,b,dis/(2*b),dis)


#print 'BF filter'
print 'BF start'
# start_time=time.time()
# for compressed_data dimension is (sample_number, sample_length/compressed_lendgth)

# data train dimension should be (number_of_ , length_of_)
bf_train = bf_ts.convert_set_to_bf(data_train)  # the result it a list and hard to convert to np array
# print len(bf_train)
bf_test = bf_ts.convert_set_to_bf(data_test)
# convert_time=time.time()-start_time
# print 'using time: ',convert_time
print 'bf lendth: ',len(bf_train)
print 'BF filter done'
# print bf_train[0]
# print bf_train[0].to01()
#


face_train =bf_ts.convert_bitarray_to_train_data(bf_train,len(bf_train),length)
face_test =bf_ts.convert_bitarray_to_train_data(bf_test,len(bf_test),length)
np.save('../data/Face_train_inputdata',face_train)
np.save('../data/Face_test_inputdata',face_test)
# # ---------



data_train = np.load('../data/Face_train_inputdata.npy')
data_test = np.load('../data/Face_test_inputdata.npy')


# print data_train.shape
# print data_test.shape
# print label_train.shape
# print label_test.shape

g=(2*b+2)*300
false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )
print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive


# ------ SVM  on only compressed data----
print "Start SVM"

d_train=data_train.tolist()
d_test=data_test.tolist()
c_train = label_train[0]
c_test = label_test[0]

start_time=time.time()

problem = svm_problem(c_train,d_train)

param = svm_parameter("-q")
param.kernel_type = LINEAR
# param.kernel_type = RBF

m = svm_train(problem, param)

p_lbl, p_acc, p_val = svm_predict(c_test,d_test,m)  #
convert_time=time.time()-start_time
print 'SVM using time: ',convert_time
print 'svm_accuracy=', p_acc
# ---------------


# ##----- KNN---------------
print 'start KNN'
start_time=time.time()
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(data_train,label_train.ravel())
predict_result=knn.predict(data_test)
# print predict_result
print 'accuracy is:',accuracy_score(label_test.ravel(),predict_result)  #
convert_time=time.time()-start_time
print 'using time: ',convert_time  #15 mins
# #--------------------------------



# #---------- SRC--------------#
# print "Start SRC"
# src= SRC()
# accuracy,right,wrong = src.run_src_classifier(data_train.T,data_test.T,label_train.T,label_test.T)
# # --------


# ----knn--
# lenth: 100000 b: 3 num_hash: 5 dis: 5.0 false_positive:  0.100925190275    accuracy is: 0.381578947368
#lenth: 8000 b: 3 num_hash: 10 dis: 5.0 false_positive:  0.010185894032 accuracy is: 0.260526315789
# lenth: 30000 b: 2 num_hash: 10 dis: 5.0 false_positive:  0.01 accuracy is: 0.318421052632
# lenth: 300000 b: 2 num_hash: 10 dis: 5.0 false_positive:  0.010185894032 accuracy is: 0.318421052632
# lenth: 300000 b: 5 num_hash: 10 dis: 10.0 false_positive:  0.010185894032 accuracy is: 0.313157894737
# lenth: 300000 b: 5 num_hash: 10 dis: 20.0 false_positive:  0.010185894032 accuracy is: 0.165789473684
# lenth: 30000 b: 5 num_hash: 10 dis: 5.0 false_positive:  0.23360244097 accuracy is: 0.315789473684
# lenth: 30000 b: 3 num_hash: 5 dis: 5.0 false_positive:  0.100925190275 accuracy is: 0.386842105263
# lenth: 30000 b: 5 num_hash: 2 dis: 5.0 false_positive:  0.399576400894 accuracy is: 0.321052631579

# --------SVM------
# lenth: 30000 b: 5 num_hash: 5 dis: 5.0 false_positive:  0.100925190275 accuracy is: 0.15
# lenth: 30000 b: 5 num_hash: 2 dis: 5.0 false_positive:  0.399576400894 accuracy is: 0.136842105263
# lenth: 3000 b: 5 num_hash: 2 dis: 5.0 false_positive:  0.902904615441 accuracy is: 0.0710526315789
# lenth: 100000 b: 5 num_hash: 2 dis: 5.0 false_positive:  0.399576400894 a/ccuracy is: 0.152631578947
# lenth: 800000 b: 5 num_hash: 2 dis: 5.0 false_positive:  0.399576400894  accuracy is: 0.157894736842