# test if data after add neighbours can be classififed (dice or jaccard)
import time
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


# load data
data_train=np.load('../data/HAR_traindata.npy') #(7352,561)
data_test=np.load('../data/HAR_testdata.npy')  #(2947 , 561)
label_train=np.load('../data/HAR_trainlabel.npy')[:3][:] #(7352,)
label_test=np.load('../data/HAR_testlabel.npy') [:3][:]#(2947,)

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

# ----

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
    g = (2 * b + 2) * 561
    num_hash = int(math.ceil(float(length / amount*(g)) * np.log(2)))
    return num_hash, dis

# -------ML on compressed and BF data--------
length = 10000
b = 10
num_hash = 10
dis = float(10)

# num_hash, dis = cal_BF_parrameter(data_train.ravel(),b,length)
# print num_hash,dis

g=(2*b+2)*561
false_positive= math.pow( 1-math.exp(-float(num_hash*g)/length) , num_hash )
print 'fp: ',false_positive

# -----BF-----
# generate the npy with the bf and data
bf_ts=BF_TS(length,num_hash,b,dis/(2*b),dis)

print 'Date pre-process'
bf_train=bf_ts.convert_set_to_bf_intermediate(data_train[:3][:])
bf_test=bf_ts.convert_set_to_bf_intermediate(data_test[:3][:])

# bf_train=bf_ts.convert_set_to_bf_intermediate(data_train[:2][:])
# data_train=np.array(list(bf_train))
# print len(data_train[0])
# print len(data_train[1])
# print data_train.shape
# print type(list(data_train[0])[6700])

# convert to np array howver array still store set
tmp_train=np.array(list(bf_train))
tmp_test=np.array(list(bf_test))


print tmp_train[0]
print list(tmp_train[0])
print len(list(tmp_train[0]))


# print tmp_train.shape

# convert set to array for classifiers
def convert_set_to_array(inputdata):
    print inputdata.shape
    a= inputdata.shape
    # outputdata = np.array((), dtype=np.str_)
    outputdata = np.empty((a[0],g), dtype=float)

    for i in range(a[0]):
        outputdata[i] = np.array(list(inputdata[i]))

    # outputdata = np.around(outputdata, decimals=5)
    return outputdata

data_train=convert_set_to_array(tmp_train)
data_test=convert_set_to_array(tmp_test)
# print data_train


# print type(str('-1.20200'))


print 'process done'
# # print bf_train[0]
# # print bf_train[0].to01()
#



# har_train =bf_ts.convert_bitarray_to_train_data(bf_train,len(bf_train),length)
# har_test =bf_ts.convert_bitarray_to_train_data(bf_test,len(bf_test),length)
# np.save('../data/HAR_train_inputdata',har_train)
# np.save('../data/HAR_test_inputdata',har_test)
# ---------

# ##----- KNN- Eu Distance  --------------
print 'Start KNN'
start_time=time.time()
knn = KNeighborsClassifier(n_neighbors=1,metric='euclidean')
knn.fit(data_train,label_train.ravel())
predict_result=knn.predict(data_test)
# print predict_result
accuracy = accuracy_score(label_test.ravel(),predict_result)
# print accuracy_score(label_test.ravel(),predict_result) # Knn accuracy= 0.
print 'Knn accuracy=',accuracy
convert_time=time.time()-start_time
print 'using time: ',convert_time  #15 mins
# #--------------------------------

# ##----- KNN-DIstance--------------
# print 'Start KNN'
start_time=time.time()
knn = KNeighborsClassifier(n_neighbors=1,metric='jaccard')
knn.fit(data_train,label_train.ravel())
predict_result=knn.predict(data_test)
# print predict_result
accuracy = accuracy_score(label_test.ravel(),predict_result)
# print accuracy_score(label_test.ravel(),predict_result) #
print 'Jaccard accuracy=',accuracy
convert_time=time.time()-start_time
# print 'using time: ',convert_time  #15 mins
# #--------------------------------

# ##----- KNN-DIstance--------------
# print 'Start KNN'
start_time=time.time()
knn = KNeighborsClassifier(n_neighbors=1,metric='dice')
knn.fit(data_train,label_train.ravel())
predict_result=knn.predict(data_test)
# print predict_result
accuracy = accuracy_score(label_test.ravel(),predict_result)
# print accuracy_score(label_test.ravel(),predict_result) #
print 'dice accuracy=',accuracy
convert_time=time.time()-start_time
# print 'using time: ',convert_time  #15 mins
# #--------------------------------