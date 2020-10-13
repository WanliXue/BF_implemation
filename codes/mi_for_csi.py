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

compressed_data = data_train


com_to_len = 80

# ---------------BF'ed data ML -----------
length = 10000
b = 10
num_hash = 22
dis = float(40)



g=(2*b+2)*com_to_len
false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )

print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive

# # generate the npy with the bf and data
bf_ts=BF_TS(length,num_hash,b,dis/(2*b),dis)

print 'BF filter'

bf_train = bf_ts.convert_set_to_bf(compressed_data)  # the result it a list and hard to convert to np array

print 'BF filter done'

csi_train =bf_ts.convert_bitarray_to_train_data(bf_train,len(bf_train),length)

np.save('../data/CSI_train_bf_for_repeatnesscheckfp66',csi_train)


print len(csi_train)
print csi_train[0]

unique_data = [list(x) for x in set(tuple(x) for x in csi_train)]
print len(unique_data)