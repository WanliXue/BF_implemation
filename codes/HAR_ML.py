# import os
import time
# import pandas
import numpy as np
import math
from svmutil import *
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# import matplotlib as mpl
# mpl.use('TkAgg')  #for mac using matplotlib
# import matplotlib.pyplot as plt

from BF_TS import BF_TS    # raw BF_TS with 2 digitis
from sklearn.ensemble import RandomForestClassifier
# from BF_TS_2digit_neigdis import BF_TS  #BF_TS in list with 3 digits



# from SRC_library import SRC

# ------load from txt and saved in npy
# name ='HAR_test_label'
# database = '..' + os.sep + 'data' + os.sep + '%s.txt' %name
# series = pandas.read_csv(database, header=-1, delimiter='\t')
# t_data = np.array(series, dtype=pandas.Series)
# data = np.array(t_data, dtype=np.int)
# print data.shape
# # print data
# test_label=np.reshape(data,(2947,))
# print test_label
# np.save('HAR_testlabel',test_label)
# ------------

# load data
data_train=np.load('../data/HAR_traindata.npy') #(7352,561)
data_test=np.load('../data/HAR_testdata.npy')  #(2947 , 561)
label_train=np.load('../data/HAR_trainlabel.npy') #(7352,)
label_test=np.load('../data/HAR_testlabel.npy') #(2947,)



# # -----------SVM---------
# print "Start SVM"
#
# d_train=data_train.tolist()
# d_test=data_test.tolist()
# c_train = label_train
# c_test = label_test
#
# start_time=time.time()
# problem = svm_problem(c_train,d_train)
#
# param = svm_parameter("-q")
# param.kernel_type = LINEAR
# # param.kernel_type = RBF
#
# m = svm_train(problem, param)
#
# p_lbl, p_acc, p_val = svm_predict(c_test,d_test,m)  #
# convert_time=time.time()-start_time
# print 'SVM using time: ',convert_time
# print 'svm_accuracy=', p_acc   #%96
# # ---------------

# # -----COMPRESS------
compress_matrix = np.load('../data/HAR_compress.npy')
print 'Compress'
compressed_data = np.matmul(compress_matrix,data_train.T).T
print compressed_data.shape  #(7352,50)
compressed_test_data=np.matmul(compress_matrix,data_test.T).T
# print compressed_test_data.shape #(2947,50)
print 'Compress done'
# # ----------
compressed_data = data_train
compressed_test_data = data_test


# # ------ SVM  on only compressed data----
# clf = svm.SVC(kernel='rbf')
# # clf = svm.SVC() #no difference in plain
# clf.fit(compressed_data,label_train)
#
# # print compressed_test_data[0].shape
# predict_result=clf.predict(compressed_test_data)
# print predict_result
#
# print accuracy_score(label_test,predict_result)   #0.92
# # ---------------


# -------ML on compressed and BF data--------
length = 10000
b = 10
num_hash = 5
dis = float(5)

g=(2*b+2)*50
false_positive= math.pow( 1-math.exp(-(float)(num_hash*g)/length) , num_hash )
print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive, 'stepdis',dis/(2*b)

# -----BF-----
# generate the npy with the bf and data
bf_ts=BF_TS(length,num_hash,b,0.5,-1000)

print 'BF start'
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

har_train =bf_ts.convert_bitarray_to_train_data(bf_train,len(bf_train),length)
har_test =bf_ts.convert_bitarray_to_train_data(bf_test,len(bf_test),length)
# np.save('../data/HAR_train_inputdata',har_train)
# np.save('../data/HAR_test_inputdata',har_test)
# ---------

#
# print 'done'


#
# har_train = np.load('../data/HAR_train_inputdata.npy')
# har_test = np.load('../data/HAR_test_inputdata.npy')
#
# ---random forest--


model = RandomForestClassifier(n_estimators=20,
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
model.fit(har_train,  label_train.ravel())
y_pred = model.predict(har_test)

accuracy = accuracy_score(label_test.ravel(),y_pred)
# print accuracy_score(label_test.ravel(),predict_result) # 0.86
print ('random tree',accuracy)



# -----------SVM---------
print "Start SVM"

d_train=har_train.tolist()
d_test=har_test.tolist()
c_train = label_train
c_test = label_test

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
print 'Start KNN'
start_time=time.time()
knn = KNeighborsClassifier(n_neighbors=1,metric='euclidean')
knn.fit(har_train,label_train.ravel())
predict_result=knn.predict(har_test)
# print predict_result
accuracy = accuracy_score(label_test.ravel(),predict_result)
# print accuracy_score(label_test.ravel(),predict_result) # 0.86
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

# # ##----- KNN-DIstance--------------
# # print 'Start KNN'
# start_time=time.time()
# knn = KNeighborsClassifier(n_neighbors=1,metric='matching')
# knn.fit(har_train,label_train.ravel())
# predict_result=knn.predict(har_test)
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
# knn.fit(har_train,label_train.ravel())
# predict_result=knn.predict(har_test)
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
# knn.fit(har_train,label_train.ravel())
# predict_result=knn.predict(har_test)
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
# knn.fit(har_train,label_train.ravel())
# predict_result=knn.predict(har_test)
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
# knn.fit(har_train,label_train.ravel())
# predict_result=knn.predict(har_test)
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
# knn.fit(har_train,label_train.ravel())
# predict_result=knn.predict(har_test)
# # print predict_result
# accuracy = accuracy_score(label_test.ravel(),predict_result)
# # print accuracy_score(label_test.ravel(),predict_result) # 0.86
# print 'sokalmichener accuracy=',accuracy
# convert_time=time.time()-start_time
# # print 'using time: ',convert_time  #15 mins
# # #--------------------------------

# # # ----SRC--------
# print 'SRC'
# src=SRC()
#
# # src on raw data
# # accuracy,right,wrong=src.run_src_classifier(data_train.T,data_test.T,label_train,label_test)
# # src on compressed
# # accuracy,right,wrong=src.run_src_classifier(compressed_data.T,compressed_test_data.T,label_train,label_test)
# # src on bf data
# accuracy,right,wrong=src.run_src_classifier_bf(har_train.T,har_test.T,label_train,label_test)
# # # -----------


# g=(2*b+2)*50
# false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )
# print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive
# print 'accuracy is: ',accuracy

# HAR SVM-----    #0.92
# length = 10000 b = 2 num_hash = 2 dis = float(5)  fp=0.4 acc = 0.80
# length = 10000 b = 2 num_hash = 5 dis = float(5)  fp=0.1 acc = 0.84
# length = 10000 b = 5 num_hash = 15 dis = float(5)  fp=0.001 acc = 0.89
# length = 10000 b = 5 num_hash = 10 dis = float(5)  fp=0.01   acc= 0.89
# length = 10000 b = 5 num_hash = 2 dis = float(5)  fp=0.4 acc = 0.87
# length = 10000 b = 5 num_hash = 5 dis = float(5)  fp=0.1 acc = 0.88
# length = 10000 b = 10 num_hash = 5 dis = float(5)  fp=0.1 acc = 0.90
# length = 10000 b = 15 num_hash = 5 dis = float(5)  fp=0.1 acc = 0.886


# knn
# length = 10000 b = 2 num_hash = 5 dis = float(5)  fp=0.1 accuracy is:  0.69
# lenth: 10000 b: 10 num_hash: 5 dis: 5.0 false_positive:  0.100925190275  acc=0.690872073295
# lenth: 10000 b: 10 num_hash: 5 dis: 20.0 false_positive:  0.100925190275 acc=0.690872073295
# lenth: 10000 b: 5 num_hash: 15 dis: 5.0 false_positive:  0.0010280132933 acc=0.690872073295

# ----- new result-----
