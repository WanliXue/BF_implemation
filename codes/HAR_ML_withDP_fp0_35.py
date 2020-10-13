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
# from SRC_library import SRC

import time
start_time = time.time()

# load data
data_train=np.load('../data/HAR_traindata.npy') #(7352,561)
data_test=np.load('../data/HAR_testdata.npy')  #(2947 , 561)
label_train=np.load('../data/HAR_trainlabel.npy') #(7352,)
label_test=np.load('../data/HAR_testlabel.npy') #(2947,)

# -----COMPRESS------
compress_matrix = np.load('../data/HAR_compress.npy')
print 'Compress'
compressed_data = np.matmul(compress_matrix,data_train.T).T
# print compressed_data.shape  #(7352,50)
compressed_test_data=np.matmul(compress_matrix,data_test.T).T
# print compressed_test_data.shape #(2947,50)
print 'Compress done'
# ----------


# -------ML on compressed and BF data--------
length = 10000
b = 10
num_hash = 20
dis = float(5)

g=(2*b+2)*50
false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )
print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive

# -----BF-----
# generate the npy with the bf and data
bf_ts=BF_TS(length,num_hash,b,dis/(2*b),dis)

print 'BF start'
bf_train = bf_ts.convert_set_to_bf(compressed_data)  # the result it a list and hard to convert to np array
bf_test = bf_ts.convert_set_to_bf(compressed_test_data)
print 'BF filter done'

#

har_train =bf_ts.convert_bitarray_to_train_data(bf_train,len(bf_train),length)
har_test =bf_ts.convert_bitarray_to_train_data(bf_test,len(bf_test),length)
print type(har_test)
# np.save('../data/HAR_train_inputdata',har_train)
# np.save('../data/HAR_test_inputdata',har_test)
# ---------
#
# har_train = np.load('../data/HAR_train_inputdata.npy')
# har_test = np.load('../data/HAR_test_inputdata.npy')
#

# -------- ADDing Laploace Noise------
loc  = 0.  # location = mean
# scale =1.  # location = mean scale=decay
epsilon = 1
sensity =1.0
print 'epsilon: ',epsilon
scale = sensity/epsilon
print 'scale: ', scale
p=0.2  #noise possibility


print 'epsilon :',epsilon,', p-value, ', p

s = np.random.laplace(loc, scale, length)
har_train=bf_ts.adding_lp_noise_to_bf_data(har_train,epsilon,length , p)

# ------------------





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

# print("--- %s seconds ---" % (time.time() - start_time))
#
# import smtplib
# subject='har'
# text='done'
# content = ('Subject: %s\n\n%s' % (subject, text))
# mail = smtplib.SMTP('smtp.gmail.com',587)
# mail.ehlo()
# mail.starttls()
# mail.login('xuewanli.lee@gmail.com','chiLLY108427')
# mail.sendmail('xuewanli.lee@gmail.com','xuewanli.lee@gmail.com',content)
# mail.close()
# print("Sent")

