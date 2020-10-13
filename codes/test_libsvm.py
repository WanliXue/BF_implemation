from svmutil import *
import scipy.io
import numpy as np

# y, x = svm_read_problem('../heart_scale')
# m = svm_train(y[:200], x[:200], '-c 4')
# p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)
#

# y, x = [1,-1], [[1,0,1], [-1,0,-1]]
# # Sparse data
# y, x = [1,-1], [{1:1, 3:1}, {1:-1,3:-1}]
# prob  = svm_problem(y, x)
# param = svm_parameter('-t 0 -c 4 -b 1')
# m = svm_train(prob, param)
#




# from svm import *
# prob = svm_problem([1,-1], [{1:1, 2:1}, {1:-1,2:-1}])
# param = svm_parameter('-c 4')
# m = libsvm.svm_train(prob, param) # m is a ctype pointer to an svm_model
# # onvert a Python-format instance to svm_nodearray, a ctypes structure
# x0, max_idx = gen_svm_nodearray({1:1, 2:1})
# label = libsvm.svm_predict(m, x0)
# print label


# # -----------------
# data=scipy.io.loadmat('../data/csi.mat')
#
# data_train=data['train_set']
# data_test = data['test_set']
# label_train = data['train_label']
# label_test = data['test_label']
#
# # data_train = np.delete(data_train,0,1)
# # data_test = np.delete(data_test,0,1)
# #
#
# label_train = label_train.T[0]
# # print label_train
#
# c_test = label_test.T[0]
#
# print data_train.shape # (150,80)
#
#
# d_train = data_train.tolist()  #turn np array to list
# d_test = data_test.tolist()
#
# param = svm_parameter("-q")
# param.kernel_type = LINEAR
# # param.kernel_type = RBF
# # param.cross_validation=1
# # param.nr_fold=10
#
#
# problem = svm_problem(label_train,d_train)
#
# m=svm_train(problem,param)
#
# pred_lbl,pred_acc,pred_val = svm_predict(c_test,d_test,m)   #90%
# print pred_acc
# # --------------------



# -----test libsvm on mnist --------
data=scipy.io.loadmat('../data/10mnist_data.mat')

# print data['test_set'].shape #(1000,784)
train_data=data['train_set']
test_data=data['test_set']
train_label=data['train_label']
test_label=data['test_label']

print train_data.shape
print test_data.shape
print train_label.shape
print test_label.shape

c_train = train_label
print len(c_train)
c_test = test_label

print len(c_test)

d_train = train_data.tolist()
d_test = test_data.tolist()
print len(d_train)
print len(d_test)

problem = svm_problem(c_train,d_train)

param = svm_parameter("-q")
param.kernel_type = LINEAR
# param.kernel_type = RBF

m = svm_train(problem, param)

p_lbl, p_acc, p_val = svm_predict(c_test,d_test,m)  #90.1%
print p_acc