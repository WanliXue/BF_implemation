import numpy as np
import math
from BF_TS_2digit_neigdis import BF_TS
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data_train=np.load('../data/HAR_traindata.npy') #(7352,561)
data_test=np.load('../data/HAR_testdata.npy')  #(2947 , 561)
label_train=np.load('../data/HAR_trainlabel.npy') #(7352,)
label_test=np.load('../data/HAR_testlabel.npy') #(2947,)


length = 10000
b = 0
num_hash = 5
# dis = float(0.5)

# g=(2*b+2)*50
# false_positive= math.pow( 1-math.exp(-(float)(num_hash*g)/length) , num_hash )
# print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive, 'stepdis',dis/(2*b)


bf_ts=BF_TS(length,num_hash,b,0,-1000)


bf_train = bf_ts.convert_set_to_bf(data_train)  # the result it a list and hard to convert to np array
bf_test = bf_ts.convert_set_to_bf(data_test)

data_train =bf_ts.convert_bitarray_to_train_data(bf_train,len(bf_train),length)
data_test =bf_ts.convert_bitarray_to_train_data(bf_test,len(bf_test),length)

random.random()


def add_rappor_noise(input,h,f):
    a, b = input.shape
    for i in range(a):
        tem_data = input[i]
        # print(tem_data.shape)
        aa=tem_data.shape[0] # first index
        for t in range(aa):
            once = random.random()
            if (once < f):
                if (once > (f / 2)):
                    input[i][t] = 1
                else:
                    input[i][t] = 0
    return input


    # retur
#

f=0.9
rapor_train = add_rappor_noise(data_train,num_hash,f)
print(rapor_train.shape)


print 'Start KNN'

knn = KNeighborsClassifier(n_neighbors=1,metric='euclidean')
knn.fit(rapor_train,label_train.ravel())
predict_result=knn.predict(data_test)
# print predict_result
accuracy = accuracy_score(label_test.ravel(),predict_result)
# print accuracy_score(label_test.ravel(),predict_result) # 0.86
print 'Knn accuracy=',accuracy

# print 'using time: ',convert_time  #15 mins
# #--------------------------------

# f=0.1  acc = 0.22
# f =0.2, Knn accuracy= 0.22
# f = 0.3, acc = 0.212
# f = 0.4, acc =0.20
# f = 0.5, acc =0.20
# f = 0.6, acc =0.19
# f = 0.7, acc =0.19
# f = 0.8, acc =0.16
# f = 0.9, acc = 0.18