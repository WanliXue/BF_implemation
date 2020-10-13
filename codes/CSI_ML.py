# import os
import time
import scipy.io
# import pandas
import numpy as np
import math
from svmutil import *
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from BF_TS_2digit_neigdis import BF_TS
# from SRC_library import SRC
from sklearn.ensemble import RandomForestClassifier

data=scipy.io.loadmat('../data/csi.mat')

data_train=data['train_set']
data_test = data['test_set']
label_train = data['train_label']
label_test = data['test_label']

# data_train = np.delete(data_train,0,1)
# data_test = np.delete(data_test,0,1)

print data_train.shape
print data_test.shape
compressed_data = data_train
compressed_test_data = data_test

print (compressed_data.shape)
print (compressed_test_data.shape)


# data_train=np.around(  (data_train-np.min(data_train))/np.ptp(data_train), decimals=4  )
# data_test=np.around(   (data_test-np.min(data_test))/np.ptp(data_test), decimals=4  )
#

# print data_train
# print data_test.shape
# print label_train.ravel()
# print label_train.ravel()

# ------ SVM  on only compressed data----
# # clf = svm.SVC(kernel='rbf')
# clf = svm.SVC() #no difference in plain
# clf.fit(data_train,label_train.ravel())
#
# # print compressed_test_data[0].shape
# predict_result=clf.predict(data_test)
# print predict_result
#
# print accuracy_score(label_test.ravel(),predict_result)  #
# # ---------------

# # # load data
# data_train=np.load('../data/CSI_traindata.npy') #(150,10800)
# data_test=np.load('../data/CSI_testdata.npy')  #(50 , 10800)
# label_train=np.load('../data/CSI_trainlabel.npy') #(150,)
# label_test=np.load('../data/CSI_testlabel.npy') #(50,)
#
# #
#
com_to_len = 80

# # # -------- SVD the data set get the compression matrix---------
# u,s,vh = np.linalg.svd(data_train.T,full_matrices= True)
# print u.shape
#
# # get the compression matrix
# compress_matrix = u[:,0:com_to_len].T
# # print compress_matrix.shape  # (com_to_len , 10800)
# np.save('CSI_compress',compress_matrix)
# # # ----------------------

# compress_matrix = np.load('../data/CSI_compress.npy') # (100,10800)
# print compress_matrix.shape

#
# ## ---- compress----
# print 'Compress'
# compressed_data = np.matmul(compress_matrix,data_train.T).T
# # print compressed_data.shape  # (150,com_to_len)
# compressed_test_data=np.matmul(compress_matrix,data_test.T).T
# # print compressed_test_data.shape  # (50,com_to_len)
# print 'Compress done'
# # #-------

# compressed_data = np.load('../data/CSI_cs_train.npy')
# compressed_test_data = np.load('../data/CSI_cs_test.npy')
# #
# label_train = np.load('../data/CSI_trainlabel.npy')
# label_test = np.load('../data/CSI_testlabel.npy')


# # ------ SVM 2 with libsvm-----
c_train = label_train
c_test = label_test
d_train = compressed_data.tolist()
d_test = compressed_test_data.tolist()

problem = svm_problem(c_train,d_train)

param = svm_parameter("-q")
param.kernel_type = LINEAR
# param.kernel_type = RBF

m = svm_train(problem, param)

p_lbl, p_acc, p_val = svm_predict(c_test,d_test,m)  #90% on compressed only
print 'SVM accuracy is ', p_acc

# # --------------------

#
# # ------ SVM  on only compressed data---- sklearn----- comment
# clf = svm.SVC(kernel='rbf')
# clf = svm.SVC() #no difference in plain
# clf.fit(compressed_data,label_train.ravel())
#
# # print compressed_test_data[0].shape
# predict_result=clf.predict(compressed_test_data)
# print predict_result
#
# print accuracy_score(label_test.ravel(),predict_result)  #
# # ---------------


# # # ----SRC--------
# print 'SRC'
# src=SRC()
#
# # src on raw data
# # accuracy,right,wrong=src.run_src_classifier(data_train.T,data_test.T,label_train,label_test)
# # src on compressed
# accuracy,right,wrong=src.run_src_classifier(compressed_data.T,compressed_test_data.T,label_train,label_test)
# # src on bf data
# # accuracy,right,wrong=src.run_src_classifier(har_train.T,har_test.T,label_train,label_test)
# # # -----------

#
# ##----- KNN---------------
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(compressed_data,label_train.ravel())
predict_result=knn.predict(compressed_test_data)
# print predict_result
print 'knn acuracy is ',accuracy_score(label_test.ravel(),predict_result) # 0.86
# #--------------------------------



# ---------------BF'ed data ML -----------
length = 10000
b =10
num_hash = 5
dis = float(50)



g=(2*b+2)*com_to_len
false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )

print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive

# # generate the npy with the bf and data
bf_ts=BF_TS(length,num_hash,b,dis/(2*b),dis)

print 'BF filter'
# print 'start'
# start_time=time.time()
bf_train = bf_ts.convert_set_to_bf(compressed_data)  # the result it a list and hard to convert to np array
# print len(bf_train)
bf_test = bf_ts.convert_set_to_bf(compressed_test_data)
# convert_time=time.time()-start_time
# print 'using time: ',convert_time
# print len(bf_train)
print 'BF filter done'
# # print bf_train[0]
# # print bf_train[0].to01()
#
csi_train =bf_ts.convert_bitarray_to_train_data(bf_train,len(bf_train),length)
csi_test =bf_ts.convert_bitarray_to_train_data(bf_test,len(bf_test),length)
# np.save('../data/CSI_train_inputdata',csi_train)
# np.save('../data/CSI_test_inputdata',csi_test)
# --------------

# print type(csi_test)


# ---train---on compressed BF----


# clf = svm.SVC()
# clf.fit(csi_train,label_train.ravel())

# ---random forest--

model = RandomForestClassifier(n_estimators=20,
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
model.fit(csi_train,  label_train.ravel())
y_pred = model.predict(csi_test)

accuracy = accuracy_score(label_test.ravel(),y_pred)
# print accuracy_score(label_test.ravel(),predict_result) # 0.86
print ('random tree',accuracy)


# -----KNN-------
print 'start KK train'
start_time=time.time()
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(csi_train,label_train.ravel())


## print compressed_test_data[0].shape
# predict_result=clf.predict(csi_test)
## print predict_result

convert_time=time.time()-start_time
print 'KNN using time: ',convert_time

predict_result=knn.predict(csi_test)

print 'knn_accuracy=', accuracy_score(label_test.ravel(),predict_result)
# ---------------

# ##----- KNN-DIstance--------------
# print 'Start KNN'
start_time=time.time()
knn = KNeighborsClassifier(n_neighbors=1,metric='jaccard')
knn.fit(csi_train,label_train.ravel())
predict_result=knn.predict(csi_test)
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
knn.fit(csi_train,label_train.ravel())
predict_result=knn.predict(csi_test)
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
knn.fit(csi_train,label_train.ravel())
predict_result=knn.predict(csi_test)
# print predict_result
accuracy = accuracy_score(label_test.ravel(),predict_result)
# print accuracy_score(label_test.ravel(),predict_result) # 0.86
print 'hamming accuracy=',accuracy
convert_time=time.time()-start_time
# print 'using time: ',convert_time  #15 mins
# #--------------------------------

# # ##----- KNN-DIstance--------------
# # print 'Start KNN'
# start_time=time.time()
# knn = KNeighborsClassifier(n_neighbors=1,metric='matching')
# knn.fit(csi_train,label_train.ravel())
# predict_result=knn.predict(csi_test)
# # print predict_result
# accuracy = accuracy_score(label_test.ravel(),predict_result)
# # print accuracy_score(label_test.ravel(),predict_result) # 0.86
# print 'matching accuracy=',accuracy
# convert_time=time.time()-start_time
# # print 'using time: ',convert_time  #15 mins
# # #--------------------------------
#
#
# # ##----- KNN-DIstance--------------
# # print 'Start KNN'
# start_time=time.time()
# knn = KNeighborsClassifier(n_neighbors=1,metric='kulsinski')
# knn.fit(csi_train,label_train.ravel())
# predict_result=knn.predict(csi_test)
# # print predict_result
# accuracy = accuracy_score(label_test.ravel(),predict_result)
# # print accuracy_score(label_test.ravel(),predict_result) # 0.86
# print 'kulsinski accuracy=',accuracy
# convert_time=time.time()-start_time
# # print 'using time: ',convert_time  #15 mins
# # #--------------------------------
#
# # ##----- KNN-DIstance--------------
# # print 'Start KNN'
# start_time=time.time()
# knn = KNeighborsClassifier(n_neighbors=1,metric='rogerstanimoto')
# knn.fit(csi_train,label_train.ravel())
# predict_result=knn.predict(csi_test)
# # print predict_result
# accuracy = accuracy_score(label_test.ravel(),predict_result)
# # print accuracy_score(label_test.ravel(),predict_result) # 0.86
# print 'rogerstanimoto accuracy=',accuracy
# convert_time=time.time()-start_time
# # print 'using time: ',convert_time  #15 mins
# # #--------------------------------
#
# # ##----- KNN-DIstance--------------
# # print 'Start KNN'
# start_time=time.time()
# knn = KNeighborsClassifier(n_neighbors=1,metric='russellrao')
# knn.fit(csi_train,label_train.ravel())
# predict_result=knn.predict(csi_test)
# # print predict_result
# accuracy = accuracy_score(label_test.ravel(),predict_result)
# # print accuracy_score(label_test.ravel(),predict_result) # 0.86
# print 'russellrao accuracy=',accuracy
# convert_time=time.time()-start_time
# # print 'using time: ',convert_time  #15 mins
# # #--------------------------------
#
#
# # ##----- KNN-DIstance--------------
# # print 'Start KNN'
# start_time=time.time()
# knn = KNeighborsClassifier(n_neighbors=1,metric='sokalsneath')
# knn.fit(csi_train,label_train.ravel())
# predict_result=knn.predict(csi_test)
# # print predict_result
# accuracy = accuracy_score(label_test.ravel(),predict_result)
# # print accuracy_score(label_test.ravel(),predict_result) # 0.86
# print 'sokalsneath accuracy=',accuracy
# convert_time=time.time()-start_time
# # print 'using time: ',convert_time  #15 mins
# # #--------------------------------
#
# # ##----- KNN-DIstance--------------
# # print 'Start KNN'
# start_time=time.time()
# knn = KNeighborsClassifier(n_neighbors=1,metric='sokalmichener')
# knn.fit(csi_train,label_train.ravel())
# predict_result=knn.predict(csi_test)
# # print predict_result
# accuracy = accuracy_score(label_test.ravel(),predict_result)
# # print accuracy_score(label_test.ravel(),predict_result) # 0.86
# print 'sokalmichener accuracy=',accuracy
# convert_time=time.time()-start_time
# # print 'using time: ',convert_time  #15 mins
# # #--------------------------------


# ------ SVM  on only compressed data---- sklearn----- comment
# clf = svm.SVC(kernel='rbf')
clf = svm.SVC() #no difference in plain
clf.fit(csi_train,label_train.ravel())

# print compressed_test_data[0].shape
predict_result=clf.predict(csi_test)
# print predict_result

print 'svm(sklearn) acc=',accuracy_score(label_test.ravel(),predict_result)  #
# ---------------
#
#
# ----SVM on compress BF data----
c_train = label_train
c_test = label_test
d_train = csi_train.tolist()
d_test = csi_test.tolist()

start_time=time.time()
problem = svm_problem(c_train,d_train)

param = svm_parameter("-q")
param.kernel_type = LINEAR
# param.kernel_type = RBF

m = svm_train(problem, param)

p_lbl, p_acc, p_val = svm_predict(c_test,d_test,m)  #90% on compressed only
convert_time=time.time()-start_time
print 'SVM using time: ',convert_time
print 'svm_accuracy=', p_acc
# -----------

# # # ----SRC--------
# print 'start SRC'
# start_time=time.time()
# src=SRC()
# accuracy,right,wrong=src.run_src_classifier(csi_train.T,csi_test.T,label_train,label_test)
# convert_time=time.time()-start_time
# print 'SRC using time: ',convert_time
# print 'SRC_accuracy=',accuracy
# # # -----------

# f= (1 -e^(-num_hash * [(2*b+2)*com_to_len] /length))^num_hash


# ------BF ed ----
# 1 change num_hash
# KNN using time:  0.0301439762115  SVM using time:  6.9133579731
# lenth: 10000 b: 10 num_hash: 12 dis: 10.0 false_positive:  0.54 knn_acc=0.76 svm_acc=0.58
# lenth: 10000 b: 10 num_hash: 10 dis: 10.0 false_positive:  0.23 knn_acc=0.8 svm_acc=0.62
# lenth: 10000 b: 10 num_hash: 8 dis: 10.0 false_positive: 0.312 KNN_acc=0.82 svm_acc=0.66
# lenth: 10000 b: 10 num_hash: 6 dis: 10.0 false_positive: 0.41 knn_accuracy= 0.84 svm_accuracy=62.0
# lenth: 10000 b: 10 num_hash: 4 dis: 10.0 false_positive: 0.41 knn_accuracy=0.8  svm_accuracy=0.64
# lenth: 10000 b: 10 num_hash: 2 dis: 10.0 false_positive:  0.39 knn_accuracy=0.82  svm_accuracy=0.66


# 2 change b

# lenth: 10000 b: 12 num_hash: 6 dis: 10.0 false_positive:0.41  knn_accuracy=0.6  svm_accuracy=0.38
# lenth: 10000 b: 10 num_hash: 6 dis: 10.0 false_positive: 0.41 knn_accuracy= 0.84 svm_accuracy=62.0
# lenth: 10000 b: 8 num_hash: 6 dis: 10.0 false_positive:  0.06 knn_accuracy= 0.62 svm_accuracy= 48.0
# lenth: 10000 b: 6 num_hash: 6 dis: 10.0 false_positive:  0.06 knn_accuracy= 0.66 svm_accuracy= 38
# lenth: 10000 b: 4 num_hash: 6 dis: 10.0 false_positive:  0.06 knn_accuracy= 0.8 svm_accuracy=72
# lenth: 10000 b: 2 num_hash: 6 dis: 10.0 false_positive:  0.06 knn_accuracy=0.82  svm_accuracy=76
# lenth: 10000 b: 1 num_hash: 6 dis: 10.0 false_positive:  0.063 knn_accuracy= 0.86 svm_accuracy= 78


#3 change dis

# lenth: 10000 b: 10 num_hash: 6 dis: 6.0 false_positive:  0.41 knn_accuracy=0.6 svm_accuracy= 50
# lenth: 10000 b: 10 num_hash: 6 dis: 8.0 false_positive: 0.41 knn_accuracy= 0.7 svm_accuracy= 62
# lenth: 10000 b: 10 num_hash: 6 dis: 10.0 false_positive: 0.41 knn_accuracy= 0.84 svm_accuracy=62.0
# lenth: 10000 b: 10 num_hash: 6 dis: 12.0 false_positive: 0.41 knn_accuracy= 0.84 svm_accuracy=72
# lenth: 10000 b: 10 num_hash: 6 dis: 14.0 false_positive: 0.41 knn_accuracy= 0.86 svm_accuracy=74

# lenth: 10000 b: 10 num_hash: 6 dis: 18.0 false_positive: 0.41 knn_accuracy= 0.86 svm_accuracy=80
# * lenth: 10000 b: 10 num_hash: 6 dis: 20.0 false_positive: 0.41 knn_accuracy= 0.9 svm_accuracy=84
# lenth: 10000 b: 10 num_hash: 6 dis: 24.0 false_positive: 0.41 knn_accuracy= 0.88 svm_accuracy=84
# lenth: 10000 b: 10 num_hash: 6 dis: 30.0 false_positive: 0.41 knn_accuracy= 0.86 svm_accuracy=84

#  b affect the svm accuracy
# lenth: 10000 b: 1 num_hash: 6 dis: 30.0 false_positive:  0.06 knn_accuracy= 0.8 svm_accuracy=80
# lenth: 10000 b: 1 num_hash: 6 dis: 20.0 false_positive:  0.06 knn_accuracy= 0.84 svm_accuracy= 82

# 4 change length

# lenth: 2000 b: 10 num_hash: 6 dis: 20.0 false_positive:  0.98 knn_accuracy= 0.18 svm_accuracy=18
# lenth: 3000 b: 10 num_hash: 6 dis: 20.0 false_positive:  0.89 knn_accuracy= 0.66 svm_accuracy=60
# lenth: 4000 b: 10 num_hash: 6 dis: 20.0 false_positive:  0.73 knn_accuracy= 0.86 svm_accuracy=80
# lenth: 6000 b: 10 num_hash: 6 dis: 20.0 false_positive:  0.41 knn_accuracy= 0.86 svm_accuracy=80
# lenth: 8000 b: 10 num_hash: 6 dis: 20.0 false_positive:  0.41 knn_accuracy= 0.86 svm_accuracy=82
# lenth: 10000 b: 10 num_hash: 6 dis: 20.0 false_positive: 0.41 knn_accuracy= 0.9 svm_accuracy=84
# lenth: 15000 b: 10 num_hash: 6 dis: 20.0 false_positive: 0.06 knn_accuracy= 0.84 svm_accuracy=84
# lenth: 20000 b: 10 num_hash: 6 dis: 20.0 false_positive: 0.06 knn_accuracy= 0.86 svm_accuracy=82


# -----------len=1000 b=4 dis=20--
# lenth: 10000 b: 4 num_hash: 2 dis: 20.0 false_positive:  0.399 knn_accuracy= 0.86 svm_accuracy=84.
# lenth: 10000 b: 4 num_hash: 4 dis: 20.0 false_positive:  0.16 knn_accuracy= 0.86 svm_accuracy=82.
# lenth: 10000 b: 4 num_hash: 6 dis: 10.0 false_positive:  0.06 knn_accuracy= 0.8 svm_accuracy=72
# lenth: 10000 b: 4 num_hash: 8 dis: 20.0 false_positive:  0.02 knn_accuracy= 0.84 svm_accuracy= 84.
# lenth: 10000 b: 4 num_hash: 10 dis: 20.0 false_positive:  0.02 knn_accuracy= 0.84 svm_accuracy= 84.
# lenth: 10000 b: 4 num_hash: 12 dis: 20.0 false_positive:  0.02 knn_accuracy= 0.88 svm_accuracy= 84.
# lenth: 10000 b: 4 num_hash: 14 dis: 20.0 false_positive:  0.02 knn_accuracy= 0.86 svm_accuracy= 84.