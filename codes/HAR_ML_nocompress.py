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

from BF_TS import BF_TS
from SRC_library import SRC

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

# # -----COMPRESS------
# compress_matrix = np.load('../data/HAR_compress.npy')
# print 'Compress'
# compressed_data = np.matmul(compress_matrix,data_train.T).T
# # print compressed_data.shape  #(7352,50)
# compressed_test_data=np.matmul(compress_matrix,data_test.T).T
# # print compressed_test_data.shape #(2947,50)
# print 'Compress done'
# # ----------

_,amount = data_train.shape

# ------ SVM  on only compressed data----
# clf = svm.SVC(kernel='rbf')
# # clf = svm.SVC() #no difference in plain
# clf.fit(compressed_data,label_train)
#
# # print compressed_test_data[0].shape
# predict_result=clf.predict(compressed_test_data)
# print predict_result
#
# print accuracy_score(label_test,predict_result)   #0.92
# ---------------

def cal_BF_parrameter(hb_array, b, length):
    dis = 0.05 * (max(hb_array) - min(hb_array))  # used to be 0.05

    # nei_list = []
    # uniq_elem_list = set(hb_array)
    # for uniq_elem in uniq_elem_list:
    #     nei_list.append(uniq_elem)
    #     rem_uniq_elem_val = uniq_elem % (dis / b)  # Convert into values within same interval
    #     if rem_uniq_elem_val >= 0.5 * (dis / b):
    #         nei_list.append(uniq_elem + ((dis / b) - rem_uniq_elem_val))
    #     else:
    #         nei_list.append(uniq_elem - rem_uniq_elem_val)
    # num_hash = int(math.ceil(float(length / len(set(nei_list))) * np.log(2)))
    num_hash = int(math.ceil(float(length / amount*(2*b+2)) * np.log(2)))
    return num_hash, dis

# -------ML on compressed and BF data--------
length = 10000
b = 5
num_hash = 2
dis = float(1)

# num_hash, dis = cal_BF_parrameter(data_train.ravel(),b,length)
# print num_hash,dis

g=(2*b+2)*561
false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )
print 'fp: ',false_positive

# -----BF-----
# generate the npy with the bf and data
bf_ts=BF_TS(length,num_hash,b,dis/(2*b),dis)

print 'BF start'
# start_time=time.time()
bf_train = bf_ts.convert_set_to_bf(data_train)  # the result it a list and hard to convert to np array
# print len(bf_train)
bf_test = bf_ts.convert_set_to_bf(data_test)
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
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(har_train,label_train.ravel())
predict_result=knn.predict(har_test)
# print predict_result
accuracy = accuracy_score(label_test.ravel(),predict_result)
# print accuracy_score(label_test.ravel(),predict_result) # 0.86
print 'Knn accuracy=',accuracy
convert_time=time.time()-start_time
print 'using time: ',convert_time  #15 mins
# #--------------------------------


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


g=(2*b+2)*561
false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )
print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive
print 'accuracy is: ',accuracy

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
