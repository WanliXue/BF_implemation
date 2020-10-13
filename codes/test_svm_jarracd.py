import scipy.io
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
import sklearn.metrics as skmetric
import math
from BF_TS import BF_TS
from svmutil import *
import time

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

com_to_len = 80

# # -------- SVD the data set get the compression matrix---------
# u,s,vh = np.linalg.svd(train_data.T,full_matrices= True)
# print u.shape
#
# # get the compression matrix
# compress_matrix = u[:,0:com_to_len].T
# print compress_matrix.shape  # (com_to_len , 784)
# np.save('MNIST_compress',compress_matrix)
# # ----------------------

# compress_matrix=np.load('../data/MNIST_compress.npy')
#
#
# print 'Compress'
# compressed_data = np.matmul(compress_matrix,train_data.T).T
# # print compressed_data.shape  #(7352,50)
# compressed_test_data=np.matmul(compress_matrix,test_data.T).T
# # print compressed_test_data.shape #(2947,50)
# print 'Compress done'
#


# ------------Put into  compressed and BF data---------
length = 10000
b=5
num_hash = 10
dis = float(5)

g=(2*b+2)*com_to_len
false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )

print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive



#
# # generate the npy with the bf and data
bf_ts=BF_TS(length,num_hash,b,dis/(2*b),dis)
#
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


csi_train =bf_ts.convert_bitarray_to_train_data(bf_train,len(bf_train),length)
csi_test =bf_ts.convert_bitarray_to_train_data(bf_test,len(bf_test),length)

def jac_kernel(X,Y):
    sigma=0.0002
    j_distance=skmetric.jaccard_similarity_score(X,Y)
    return math.exp(-sigma*(j_distance**2))


def my_kernel(X, Y):
    """
    We create a custom kernel:

                 (2  0)
    k(X, Y) = X  (    ) Y.T
                 (0  1)
    """
    M = np.array([[2, 0], [0, 1.0]])
    return np.dot(np.dot(X, M), Y.T)


# ------ SVM  on only compressed data---sklearn-
# clf = svm.SVC(kernel=jac_kernel)
clf = svm.SVC() #no difference in plain
clf.fit(csi_train.tolist(),label_train)

# print compressed_test_data[0].shape
predict_result=clf.predict(csi_test.tolist())
# print predict_result

print accuracy_score(label_test,predict_result)
print skmetric.jaccard_similarity_score(label_test,predict_result)

# ----SVM on compress BF data----
c_train = label_train
c_test = label_test
d_train = csi_train.tolist()
d_test = csi_test.tolist()

start_time=time.time()
problem = svm_problem(c_train,d_train)

param = svm_parameter("-q")
# param.kernel_type = LINEAR
# param.kernel_type = RBF

# m = svm_train(problem,param)

K=jac_kernel(csi_train,csi_train)
KK=jac_kernel(csi_test,csi_train)


m = svm_train(problem,K, '-t 4')

p_lbl, p_acc, p_val = svm_predict(c_test,KK,d_test,m)  #90% on compressed only
convert_time=time.time()-start_time
print 'SVM using time: ',convert_time
print 'svm_accuracy=', p_acc
# -----------