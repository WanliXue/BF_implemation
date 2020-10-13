import time
import numpy as np
import math
from svmutil import *
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import scipy.io

# # -------load data____CSI------
# # data_train=np.load('../data/CSI_traindata.npy') #(150,10800)
# # data_test=np.load('../data/CSI_testdata.npy')  #(50 , 10800)
# # label_train=np.load('../data/CSI_trainlabel.npy') #(150,)
# # label_test=np.load('../data/CSI_testlabel.npy') #(50,)
# #
#
# data=scipy.io.loadmat('../data/csi.mat')
# data_train=data['train_set']
# data_test = data['test_set']
# label_train = data['train_label']
# label_test = data['test_label']
#
# # data_train = np.delete(data_train,0,1)
# # data_test = np.delete(data_test,0,1)
#
# print data_train.shape
# print data_test.shape
# compressed_data = data_train
# compressed_test_data = data_test
#
#
#
# # print data_train.shape
# # print data_test.shape
# # compress_matrix = np.load('../data/CSI_compress.npy') # (100,10800)
# # # print compress_matrix.shape
# #
# # print 'Compress'
# # compressed_data = np.matmul(compress_matrix,data_train.T).T
# # # print compressed_data.shape  # (150,com_to_len)
# # compressed_test_data=np.matmul(compress_matrix,data_test.T).T
# # # print compressed_test_data.shape  # (50,com_to_len)
# # print 'Compress done'
# # #-------



## ------load data-----MNIST-----
data=scipy.io.loadmat('../data/10mnist_data.mat')
# print data['test_set'].shape #(1000,784)
train_data=data['train_set']
test_data=data['test_set']
train_label=data['train_label']
test_label=data['test_label']
label_train=np.ravel(train_label)
label_test=np.ravel(test_label)

# print 'Compress'
# compress_matrix=np.load('../data/MNIST_compress.npy')
# compressed_data = np.matmul(compress_matrix,train_data.T).T
# # print compressed_data.shape  #(7352,50)
# compressed_test_data=np.matmul(compress_matrix,test_data.T).T
# # print compressed_test_data.shape #(2947,50)
# print 'Compress done'
##---------------------------------





# # --------load data----- HAR----
# data_train=np.load('../data/HAR_traindata.npy') #(7352,561)
# data_test=np.load('../data/HAR_testdata.npy')  #(2947 , 561)
# label_train=np.load('../data/HAR_trainlabel.npy') #(7352,)
# label_test=np.load('../data/HAR_testlabel.npy') #(2947,)
#
# # -----COMPRESS------
# compress_matrix = np.load('../data/HAR_compress.npy')
# print 'Compress'
# compressed_data = np.matmul(compress_matrix,data_train.T).T
# # print compressed_data.shape  #(7352,50)
# compressed_test_data=np.matmul(compress_matrix,data_test.T).T
# # print compressed_test_data.shape #(2947,50)
# print 'Compress done'
# # print compressed_data.shape  # (7352,50)
# # print compressed_test_data.shape  #(2947,50)
# # ----------



# # ##----- KNN- Uclidean Distance  --------------
# print 'Start KNN'
# start_time=time.time()
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(data_train,label_train.ravel())
# predict_result=knn.predict(data_test)
# # print predict_result
# accuracy = accuracy_score(label_test.ravel(),predict_result)
# # print accuracy_score(label_test.ravel(),predict_result) # Knn accuracy= 0.878520529352
# print 'Knn accuracy=',accuracy
# convert_time=time.time()-start_time
# print 'using time: ',convert_time  #15 mins
# # #--------------------------------



har_train = train_data
har_test = test_data



# ##----- KNN- Jaccard Distance  --------------
print 'Start KNN'
start_time=time.time()
knn = KNeighborsClassifier(n_neighbors=1,metric='euclidean')
knn.fit(har_train,label_train.ravel())
predict_result=knn.predict(har_test)
# print predict_result
accuracy = accuracy_score(label_test.ravel(),predict_result)
# print accuracy_score(label_test.ravel(),predict_result) # Knn accuracy= 0.1778
print 'Knn accuracy=',accuracy
convert_time=time.time()-start_time
print 'using time: ',convert_time  #15 mins
# #--------------------------------

# ##----- KNN-DIstance--------------
# print 'Start KNN'
start_time=time.time()
knn = KNeighborsClassifier(n_neighbors=1,metric='jaccard')
knn.fit(har_train,label_train.ravel())
predict_result=knn.predict(har_test)
# print predict_result
accuracy = accuracy_score(label_test.ravel(),predict_result)
# print accuracy_score(label_test.ravel(),predict_result) # 0.86
print 'Jaccard accuracy=',accuracy
convert_time=time.time()-start_time
# print 'using time: ',convert_time  #15 mins
# #--------------------------------
# ##----- KNN-DIstance--------------
# print 'Start KNN'
start_time=time.time()
knn = KNeighborsClassifier(n_neighbors=1,metric='dice')
knn.fit(har_train,label_train.ravel())
predict_result=knn.predict(har_test)
# print predict_result
accuracy = accuracy_score(label_test.ravel(),predict_result)
# print accuracy_score(label_test.ravel(),predict_result) # 0.86
print 'dice accuracy=',accuracy
convert_time=time.time()-start_time
# print 'using time: ',convert_time  #15 mins
# #--------------------------------

# ##----- KNN-DIstance--------------
# print 'Start KNN'
start_time=time.time()
knn = KNeighborsClassifier(n_neighbors=1,metric='hamming')
knn.fit(har_train,label_train.ravel())
predict_result=knn.predict(har_test)
# print predict_result
accuracy = accuracy_score(label_test.ravel(),predict_result)
# print accuracy_score(label_test.ravel(),predict_result) # 0.86
print 'hamming accuracy=',accuracy
convert_time=time.time()-start_time
# print 'using time: ',convert_time  #15 mins
# #--------------------------------

# ##----- KNN-DIstance--------------
# print 'Start KNN'
start_time=time.time()
knn = KNeighborsClassifier(n_neighbors=1,metric='matching')
knn.fit(har_train,label_train.ravel())
predict_result=knn.predict(har_test)
# print predict_result
accuracy = accuracy_score(label_test.ravel(),predict_result)
# print accuracy_score(label_test.ravel(),predict_result) # 0.86
print 'matching accuracy=',accuracy
convert_time=time.time()-start_time
# print 'using time: ',convert_time  #15 mins
# #--------------------------------


# ##----- KNN-DIstance--------------
# print 'Start KNN'
start_time=time.time()
knn = KNeighborsClassifier(n_neighbors=1,metric='kulsinski')
knn.fit(har_train,label_train.ravel())
predict_result=knn.predict(har_test)
# print predict_result
accuracy = accuracy_score(label_test.ravel(),predict_result)
# print accuracy_score(label_test.ravel(),predict_result) # 0.86
print 'kulsinski accuracy=',accuracy
convert_time=time.time()-start_time
# print 'using time: ',convert_time  #15 mins
# #--------------------------------

# ##----- KNN-DIstance--------------
# print 'Start KNN'
start_time=time.time()
knn = KNeighborsClassifier(n_neighbors=1,metric='rogerstanimoto')
knn.fit(har_train,label_train.ravel())
predict_result=knn.predict(har_test)
# print predict_result
accuracy = accuracy_score(label_test.ravel(),predict_result)
# print accuracy_score(label_test.ravel(),predict_result) # 0.86
print 'rogerstanimoto accuracy=',accuracy
convert_time=time.time()-start_time
# print 'using time: ',convert_time  #15 mins
# #--------------------------------

# ##----- KNN-DIstance--------------
# print 'Start KNN'
start_time=time.time()
knn = KNeighborsClassifier(n_neighbors=1,metric='russellrao')
knn.fit(har_train,label_train.ravel())
predict_result=knn.predict(har_test)
# print predict_result
accuracy = accuracy_score(label_test.ravel(),predict_result)
# print accuracy_score(label_test.ravel(),predict_result) # 0.86
print 'russellrao accuracy=',accuracy
convert_time=time.time()-start_time
# print 'using time: ',convert_time  #15 mins
# #--------------------------------


# ##----- KNN-DIstance--------------
# print 'Start KNN'
start_time=time.time()
knn = KNeighborsClassifier(n_neighbors=1,metric='sokalsneath')
knn.fit(har_train,label_train.ravel())
predict_result=knn.predict(har_test)
# print predict_result
accuracy = accuracy_score(label_test.ravel(),predict_result)
# print accuracy_score(label_test.ravel(),predict_result) # 0.86
print 'sokalsneath accuracy=',accuracy
convert_time=time.time()-start_time
# print 'using time: ',convert_time  #15 mins
# #--------------------------------

# ##----- KNN-DIstance--------------
# print 'Start KNN'
start_time=time.time()
knn = KNeighborsClassifier(n_neighbors=1,metric='sokalmichener')
knn.fit(har_train,label_train.ravel())
predict_result=knn.predict(har_test)
# print predict_result
accuracy = accuracy_score(label_test.ravel(),predict_result)
# print accuracy_score(label_test.ravel(),predict_result) # 0.86
print 'sokalmichener accuracy=',accuracy
convert_time=time.time()-start_time
# print 'using time: ',convert_time  #15 mins
# #--------------------------------

##------ HAR-------
# no compressed
# Knn accuracy= 0.878520529352
# using time:  15.9028048515
# Jaccard accuracy= 0.177807940278
# dice accuracy= 0.177807940278
# hamming accuracy= 0.630132337971
# matching accuracy= 0.177807940278
# kulsinski accuracy= 0.177807940278
# rogerstanimoto accuracy= 0.177807940278
# russellrao accuracy= 0.17984390906
# sokalsneath accuracy= 0.177807940278
# sokalmichener accuracy= 0.177807940278

# ----- after compress---
# Knn accuracy= 0.857821513403
# using time:  0.720567941666
# Jaccard accuracy= 0.180522565321
# dice accuracy= 0.180522565321
# hamming accuracy= 0.180522565321
# matching accuracy= 0.180522565321
# kulsinski accuracy= 0.180522565321
# rogerstanimoto accuracy= 0.180522565321
# russellrao accuracy= 0.180522565321
# sokalsneath accuracy= 0.180522565321
# sokalmichener accuracy= 0.180522565321

#-------MNIST-----
# after compressed
# Knn accuracy= 0.933
# using time:  0.755842924118
# Jaccard accuracy= 0.105
# dice accuracy= 0.105
# hamming accuracy= 0.105
# matching accuracy= 0.105
# kulsinski accuracy= 0.105
# rogerstanimoto accuracy= 0.105
# russellrao accuracy= 0.105
# sokalsneath accuracy= 0.105
# sokalmichener accuracy= 0.105

#no compress
# Knn accuracy= 0.927
# using time:  9.54935193062
# Jaccard accuracy= 0.935
# dice accuracy= 0.935
# hamming accuracy= 0.729
# matching accuracy= 0.93
# kulsinski accuracy= 0.857
# rogerstanimoto accuracy= 0.93
# russellrao accuracy= 0.684
# sokalsneath accuracy= 0.935
# sokalmichener accuracy= 0.93
# ----------------

# -------CSI----
# ----after compress--
# Knn accuracy= 0.92
# using time:  0.00444412231445
# Jaccard accuracy= 0.04
# dice accuracy= 0.04
# hamming accuracy= 0.04
# matching accuracy= 0.04
# kulsinski accuracy= 0.04
# rogerstanimoto accuracy= 0.04
# russellrao accuracy= 0.04
# sokalsneath accuracy= 0.04
# sokalmichener accuracy= 0.04

#--no compress
# Start KNN
# Knn accuracy= 0.32
# using time:  0.00612592697144
# Jaccard accuracy= 0.06
# dice accuracy= 0.06
# hamming accuracy= 0.06
# matching accuracy= 0.06
# kulsinski accuracy= 0.06
# rogerstanimoto accuracy= 0.06
# russellrao accuracy= 0.06
# sokalsneath accuracy= 0.06
# sokalmichener accuracy= 0.06
