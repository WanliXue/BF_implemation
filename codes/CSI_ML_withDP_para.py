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

from BF_TS import BF_TS
# from SRC_library import SRC


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


# #
#
com_to_len = 80



# # # ------ SVM 2 with libsvm-----
# c_train = label_train
# c_test = label_test
# d_train = compressed_data.tolist()
# d_test = compressed_test_data.tolist()
#
# problem = svm_problem(c_train,d_train)
#
# param = svm_parameter("-q")
# param.kernel_type = LINEAR
# # param.kernel_type = RBF
#
# m = svm_train(problem, param)
#
# p_lbl, p_acc, p_val = svm_predict(c_test,d_test,m)  #90% on compressed only
# print 'SVM accuracy is ', p_acc
#
# # # --------------------
#
#
# #
# # ##----- KNN---------------
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(compressed_data,label_train.ravel())
# predict_result=knn.predict(compressed_test_data)
# # print predict_result
# print 'knn acuracy is ',accuracy_score(label_test.ravel(),predict_result) # 0.86
# # #--------------------------------



# ---------------BF'ed data ML -----------
length = 10000
b =10
num_hash = 5
dis = float(35)



g=(2*b+2)*com_to_len
false_positive= math.pow( 1-math.exp(-(float)(num_hash*g)/length) , num_hash )

print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive

# # generate the npy with the bf and data
bf_ts=BF_TS(length,num_hash,b,dis/(2*b),dis)

# print 'BF filter'
# print 'start'
# start_time=time.time()
bf_train = bf_ts.convert_set_to_bf(compressed_data)  # the result it a list and hard to convert to np array
# print len(bf_train)
bf_test = bf_ts.convert_set_to_bf(compressed_test_data)
# convert_time=time.time()-start_time
# print 'using time: ',convert_time
# print len(bf_train)
# print 'BF filter done'
# # print bf_train[0]
# # print bf_train[0].to01()
#
csi_train =bf_ts.convert_bitarray_to_train_data(bf_train,len(bf_train),length)
csi_test =bf_ts.convert_bitarray_to_train_data(bf_test,len(bf_test),length)
# np.save('../data/CSI_train_inputdata',csi_train)
# np.save('../data/CSI_test_inputdata',csi_test)
# --------------

# print type(csi_test)



# -------- ADDing Laploace Noise------
loc  = 0.  # location = mean
# scale =1.  # location = mean scale=decay
epsilon = 1
sensity =1.0
print 'epsilon: ',epsilon
scale = sensity/epsilon
print 'scale: ', scale
p=0.15  #noise possibility


print 'epsilon :',epsilon,', p-value, ', p


csi_train=bf_ts.adding_lp_noise_to_bf_data(csi_train,epsilon,length , p)

# ------------------





# ---train---on compressed BF----


# clf = svm.SVC()
# clf.fit(csi_train,label_train.ravel())


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


# # ------ SVM  on only compressed data---- sklearn----- comment
# # clf = svm.SVC(kernel='rbf')
# clf = svm.SVC() #no difference in plain
# clf.fit(csi_train,label_train.ravel())
#
# # print compressed_test_data[0].shape
# predict_result=clf.predict(csi_test)
# # print predict_result
#
# print 'svm(sklearn) acc=',accuracy_score(label_test.ravel(),predict_result)  #
# # ---------------
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

