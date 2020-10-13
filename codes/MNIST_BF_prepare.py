from BF_TS import BF_TS
import scipy.io
import numpy as np
#from svmutil import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
import math
# from SRC_library import SRC

curr_time_tuple = time.localtime()

# ---- Small dataset test 10%---
# data=scipy.io.loadmat('./data/10mnist_data.mat')

# # print data['test_set'].shape #(1000,784)
# train_data=data['train_set']
# test_data=data['test_set']
# train_label=data['train_label']
# test_label=data['test_label']



# train_label=np.ravel(train_label)
# test_label=np.ravel(test_label)

# np.save('./data/mnist_train_label',train_label)
# np.save('./data/mnist_test_label',test_label)
# ------



# train_data=np.load('/home/wanli/bf_python_run/data/mnist_train_inputdata_full.npy')
# print (train_data.shape)
# test_data=np.load('/home/wanli/bf_python_run/data/mnist_test_inputdata_full.npy')
train_label=np.load('../data/mnist_train_label_full.npy')
test_label=np.load('../data/mnist_test_label_full.npy')
#train_label=np.ravel(train_label)
#test_label=np.ravel(test_label)
print (test_label.shape)

# # ##----- KNN-DIstance--------------
# # print 'Start KNN'
# start_time=time.time()
# knn = KNeighborsClassifier(n_neighbors=1,metric='jaccard')
# knn.fit(train_data,train_label.ravel())
# predict_result=knn.predict(test_data)
# # print predict_result
# accuracy = accuracy_score(test_label.ravel(),predict_result)
# # print accuracy_score(label_test.ravel(),predict_result) # 0.86
# print 'Jaccard accuracy=',accuracy
# convert_time=time.time()-start_time
# # print 'using time: ',convert_time  #15 mins
# # #--------------------------------

com_to_len = 60

# # -------- SVD the data set get the compression matrix---------
# u,s,vh = np.linalg.svd(train_data.T,full_matrices= True)
# print u.shape
#
# # get the compression matrix
# compress_matrix = u[:,0:com_to_len].T
# print compress_matrix.shape  # (com_to_len , 784)
# np.save('MNIST_compress',compress_matrix)
# # ----------------------


#-----------------------
## compress

#compress_matrix=np.load('./data/MNIST_compress.npy')
#
#
#print ('Compress')
#compressed_data = np.matmul(compress_matrix,train_data).T
#print (compressed_data.shape)  #(7352,60)
#compressed_test_data=np.matmul(compress_matrix,test_data).T
## print compressed_test_data.shape #(2947,60)
#print ('Compress done')
#
#np.save('./data/compressed_train_data_full',compressed_data)
#np.save('./data/compressed_test_data_full',compressed_test_data)
##--------------------

compressed_data = np.load('../data/compressed_train_data_full.npy')
print(compressed_data.shape,"compressed data")
compressed_test_data = np.load('../data/compressed_test_data_full.npy')
print ('Compress done')

# # ------ SVM  on only compressed data---sklearn-
# # clf = svm.SVC(kernel='rbf')
# clf = svm.SVC() #no difference in plain
# clf.fit(compressed_data,train_label)
#
# # print compressed_test_data[0].shape
# predict_result=clf.predict(compressed_test_data)
# # print predict_result
#
# print accuracy_score(test_label,predict_result)
# # ---------------


# # ------SVM on cs daata witl libsvm-------
# c_train = train_label
# c_test = test_label
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
# # ---------------------

## ------------Put into  compressed and BF data---------
length = 10000
b=10
num_hash = 5
dis = float(5)

g=(2*b+2)*com_to_len
false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )

# print ('lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive)

false_positive= math.pow( 1-math.exp(-(float)(num_hash*com_to_len)/length) , num_hash )
print ('lenth:',length,'num_hash:',num_hash,'false_positive: ', false_positive)


## # generate the npy with the bf and data
bf_ts=BF_TS(length,num_hash,b,dis/(2*b),dis)
##


#  BF without neighbours
#---------------------
print ('BF filter')
# print 'start'
# start_time=time.time()
 # the result it a list and hard to convert to np array
bf_train = bf_ts.covert_set_to_bf_noneighbour(compressed_data)
# print len(bf_train)
bf_test = bf_ts.covert_set_to_bf_noneighbour(compressed_test_data)
# convert_time=time.time()-start_time
# print 'using time: ',convert_time
# print len(bf_train)
print ('BF filter done')



mnist_train =bf_ts.convert_bitarray_to_train_data(bf_train,len(bf_train),length)
mnist_test =bf_ts.convert_bitarray_to_train_data(bf_test,len(bf_test),length)

np.save('./data/full_minist_train_BFed_noneighbour',mnist_train)
np.save('./data/full_minist_test_BFed_noneighbour',mnist_test)
print("data file prepared and saved")

#---------------------

# #---------------------
# print ('BF filter')
# # print 'start'
# # start_time=time.time()
# bf_train = bf_ts.convert_set_to_bf(compressed_data)  # the result it a list and hard to convert to np array
# # print len(bf_train)
# bf_test = bf_ts.convert_set_to_bf(compressed_test_data)
# # convert_time=time.time()-start_time
# # print 'using time: ',convert_time
# # print len(bf_train)
# print ('BF filter done')



# mnist_train =bf_ts.convert_bitarray_to_train_data(bf_train,len(bf_train),length)
# mnist_test =bf_ts.convert_bitarray_to_train_data(bf_test,len(bf_test),length)

# np.save('./data/full_minist_train_BFed_fp02',mnist_train)
# np.save('./data/full_minist_test_BFed_fp02',mnist_test)

# #---------------------
print ('BF filter done')


# mnist_train = np.load('./data/full_minist_train_BFed_fp02.npy')
# mnist_test = np.load('./data/full_minist_test_BFed_fp02.npy')


# # -------- ADDing Laploace Noise------
# loc  = 0.  # location = mean
# # scale =1.  # location = mean scale=decay
# epsilon = 1
# sensity =1.0
# # print ('epsilon: ',epsilon)
# scale = sensity/epsilon
# print ('scale: ', scale)
# p=0.4  #noise possibility

# print ('epsilon :',epsilon,', p-value, ', p)
# mnist_train=bf_ts.adding_lp_noise_to_bf_data(mnist_train,epsilon,length , p)

# # ------------------
# print ('add noise done')


# np.save('./data/mnist_train_inputdata_BF_full_fp02_p04',mnist_train)
# np.save('./data/mnist_test_inputdata_BF_full_fp02',mnist_test)


#
# # ----------------------------------

# # --------------



## ##----- KNN---------------
#knn = KNeighborsClassifier(n_neighbors=1)
#knn.fit(mnist_train,train_label.ravel())
#predict_result=knn.predict(mnist_test)
## print predict_result
#print 'Mnist KNN accuracy: ',accuracy_score(test_label.ravel(),predict_result)
## #--------------------------------
#
#
## ##----- KNN-DIstance--------------
## print 'Start KNN'
#start_time=time.time()
#knn = KNeighborsClassifier(n_neighbors=1,metric='jaccard')
#knn.fit(mnist_train,train_label.ravel())
#predict_result=knn.predict(mnist_test)
## print predict_result
#accuracy = accuracy_score(test_label.ravel(),predict_result)
## print accuracy_score(label_test.ravel(),predict_result) # 0.86
#print 'Jaccard accuracy=',accuracy
#convert_time=time.time()-start_time
## print 'using time: ',convert_time  #15 mins
## #--------------------------------
## ##----- KNN-DIstance--------------
## print 'Start KNN'
#start_time=time.time()
#knn = KNeighborsClassifier(n_neighbors=1,metric='dice')
#knn.fit(mnist_train,train_label.ravel())
#predict_result=knn.predict(mnist_test)
## print predict_result
#accuracy = accuracy_score(test_label.ravel(),predict_result)
## print accuracy_score(label_test.ravel(),predict_result) # 0.86
#print 'dice accuracy=',accuracy
#convert_time=time.time()-start_time
## print 'using time: ',convert_time  #15 mins
## #--------------------------------
## ##----- KNN-DIstance--------------
## print 'Start KNN'
#start_time=time.time()
#knn = KNeighborsClassifier(n_neighbors=1,metric='hamming')
#knn.fit(mnist_train,train_label.ravel())
#predict_result=knn.predict(mnist_test)
## print predict_result
#accuracy = accuracy_score(test_label.ravel(),predict_result)
## print accuracy_score(label_test.ravel(),predict_result) # 0.86
#print 'hamming accuracy=',accuracy
#convert_time=time.time()-start_time
## print 'using time: ',convert_time  #15 mins
## #--------------------------------
## # ##----- KNN-DIstance--------------
## # print 'Start KNN'
## start_time=time.time()
## knn = KNeighborsClassifier(n_neighbors=1,metric='matching')
## knn.fit(mnist_train,train_label.ravel())
## predict_result=knn.predict(mnist_test)
## # print predict_result
## accuracy = accuracy_score(test_label.ravel(),predict_result)
## # print accuracy_score(label_test.ravel(),predict_result) # 0.86
## print 'matching accuracy=',accuracy
## convert_time=time.time()-start_time
## # print 'using time: ',convert_time  #15 mins
## # #--------------------------------
## # ##----- KNN-DIstance--------------
## # print 'Start KNN'
## start_time=time.time()
## knn = KNeighborsClassifier(n_neighbors=1,metric='kulsinski')
## knn.fit(mnist_train,train_label.ravel())
## predict_result=knn.predict(mnist_test)
## # print predict_result
## accuracy = accuracy_score(test_label.ravel(),predict_result)
## # print accuracy_score(label_test.ravel(),predict_result) # 0.86
## print 'kulsinski accuracy=',accuracy
## convert_time=time.time()-start_time
## # print 'using time: ',convert_time  #15 mins
## # #--------------------------------
##
## # ##----- KNN-DIstance--------------
## # print 'Start KNN'
## start_time=time.time()
## knn = KNeighborsClassifier(n_neighbors=1,metric='rogerstanimoto')
## knn.fit(mnist_train,train_label.ravel())
## predict_result=knn.predict(mnist_test)
## # print predict_result
## accuracy = accuracy_score(test_label.ravel(),predict_result)
## # print accuracy_score(label_test.ravel(),predict_result) # 0.86
## print 'rogerstanimoto accuracy=',accuracy
## convert_time=time.time()-start_time
## # print 'using time: ',convert_time  #15 mins
## # #-----------------------------------------------------------
##
## # ##----- KNN-DIstance--------------
## # print 'Start KNN'
## start_time=time.time()
## knn = KNeighborsClassifier(n_neighbors=1,metric='russellrao')
## knn.fit(mnist_train,train_label.ravel())
## predict_result=knn.predict(mnist_test)
## # print predict_result
## accuracy = accuracy_score(test_label.ravel(),predict_result)
## # print accuracy_score(label_test.ravel(),predict_result) # 0.86
## print 'russellrao accuracy=',accuracy
## convert_time=time.time()-start_time
## # print 'using time: ',convert_time  #15 mins
## # #--------------------------------
## # ##----- KNN-DIstance--------------
## # print 'Start KNN'
## start_time=time.time()
## knn = KNeighborsClassifier(n_neighbors=1,metric='sokalsneath')
## knn.fit(mnist_train,train_label.ravel())
## predict_result=knn.predict(mnist_test)
## # print predict_result
## accuracy = accuracy_score(test_label.ravel(),predict_result)
## # print accuracy_score(label_test.ravel(),predict_result) # 0.86
## print 'sokalsneath accuracy=',accuracy
## convert_time=time.time()-start_time
## # print 'using time: ',convert_time  #15 mins
## # #--------------------------------
##
## # ##----- KNN-DIstance--------------
## # print 'Start KNN'
## start_time=time.time()
## knn = KNeighborsClassifier(n_neighbors=1,metric='sokalmichener')
## knn.fit(mnist_train,train_label.ravel())
## predict_result=knn.predict(mnist_test)
## # print predict_result
## accuracy = accuracy_score(test_label.ravel(),predict_result)
## # print accuracy_score(label_test.ravel(),predict_result) # 0.86
## print 'Jaccard accuracy=',accuracy
## convert_time=time.time()-start_time
## # print 'using time: ',convert_time  #15 mins
## # #--------------------------------
##
#
## -----------SVM-----------
#
#c_train = train_label
#c_test = test_label
#d_train = mnist_train.tolist()
#d_test = mnist_test.tolist()
#
#problem = svm_problem(c_train,d_train)
#
#param = svm_parameter("-q")
#param.kernel_type = LINEAR
## param.kernel_type = RBF
#
#m = svm_train(problem, param)
#
#p_lbl, p_acc, p_val = svm_predict(c_test,d_test,m)
#print 'Mnist SVM accuracy is ', p_acc
#
## # ------- SRC----
## print 'start train'
## start_time=time.time()
## src = SRC()
## accuracy,right,wrong = src.run_src_classifier(mnist_train,mnist_test,train_label,test_label)
## convert_time=time.time()-start_time
## print 'using time: ',convert_time  #
##
## # --------
##
#
#
#
#
## ------for knn--------
## lenth: 10000 b: 10 num_hash: 2 dis: 5.0 acc = 0.932  fp=0.39
#
#
## -----------for SVM ----------
## MNIST 10%   acc= #0.945
## length = 10000 b=2  num_hash = 5 dis = float(5)  fp=0.1  acc=0.869
## length = 10000 b=5  num_hash = 15 dis = float(5)  fp=0.11  acc=0.899
## length = 10000 b=10  num_hash = 5 dis = float(5)  fp=0.1  acc=0.905
## length = 10000 b=10  num_hash = 2 dis = float(5)  fp=0.39  acc=0.896
## length = 10000 b=15  num_hash = 10 dis = float(5)  fp=0.2336  acc=0.826
