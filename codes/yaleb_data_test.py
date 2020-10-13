import scipy.io as sio
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

# from BF_TS import BF_TS    # raw BF_TS with 2 digitis

from BF_TS_2digit_neigdis import BF_TS

# str = '/Users/wanli/Dropbox/CODE/Eigenface/yaleb_svd_data.mat'
str = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/yaleb_svd_data.mat'

load_mat = sio.loadmat(str)

print(load_mat['A'].shape)
print(load_mat['y'].shape)
print(load_mat['testlabels'].shape)
print(load_mat['trainlabels'].shape)

train_data =load_mat['A']
test_data = load_mat['y']
train_label = load_mat['trainlabels']
test_label = load_mat['testlabels']


 # -----------SVM---------
print "Start SVM"

d_train=train_data.T.tolist()
d_test=test_data.T.tolist()
c_train = train_label
c_test = test_label

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


compressed_data = train_data.T
compressed_test_data = test_data.T

# -------ML on compressed and BF data--------
length = 10000
b = 5
num_hash = 1
dis = float(5)

g=(2*b+1)*80
false_positive= math.pow( 1-math.exp(-(float)(num_hash*g)/length) , num_hash )
print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive, 'stepdis: ',dis/(2*b)

# -----BF-----
# generate the npy with the bf and data
bf_ts=BF_TS(length,num_hash,b,dis/(2*b),dis)

print 'BF start'
# start_time=time.time()
bf_train = bf_ts.convert_set_to_bf(compressed_data)  # the result it a list and hard to convert to np array
# print len(bf_train)
# bf_test = bf_ts.convert_set_to_bf(compressed_test_data)
# convert_time=time.time()-start_time
# print 'using time: ',convert_time
# print len(bf_train)
print 'BF filter done'
# # print bf_train[0]
# # print bf_train[0].to01()
#

har_train =bf_ts.convert_bitarray_to_train_data(bf_train,len(bf_train),length)
# har_test =bf_ts.convert_bitarray_to_train_data(bf_test,len(bf_test),length)
# np.save('../data/HAR_train_inputdata',har_train)
# np.save('../data/HAR_test_inputdata',har_test)
# ---------




# -----------SVM---------
print "Start SVM"

d_train=har_train.tolist()
d_test=har_test.tolist()
c_train = train_label
c_test = test_label

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
