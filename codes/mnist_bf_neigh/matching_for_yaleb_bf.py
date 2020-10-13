import scipy.io as sio
# import os
import time
# import pandas
import numpy as np
import math
# from svmutil import *
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# import matplotlib as mpl
# mpl.use('TkAgg')  #for mac using matplotlib
import matplotlib.pyplot as plt

# from BF_TS import BF_TS    # raw BF_TS with 2 digitis

from codes.BF_TS_2digit_neigdis import BF_TS


str = '/Users/wanli/Dropbox/CODE/Eigenface/yaleb_svd_data.mat'

load_mat = sio.loadmat(str)

print(load_mat['A'].shape)
print(load_mat['y'].shape)
print(load_mat['testlabels'].shape)
print(load_mat['trainlabels'].shape)

train_data =load_mat['A']
test_data = load_mat['y']
train_label = load_mat['trainlabels']
test_label = load_mat['testlabels']

compressed_data = train_data.T[0:1]
print(compressed_data.shape)



length = 10000
b=5
num_hash = 1
dis = float(5)
com_to_len = 80

# g=(2*b+1)*com_to_len
false_positive= math.pow( 1-math.exp(-float(num_hash*2*b*com_to_len)/length) , num_hash )
print ('lenth:',length,'num_hash:',num_hash,'total distance:',dis,\
       'step dis:',dis/(2*b),'neighbours:',b, 'false_positive: ', false_positive)
bf_ts=BF_TS(length,num_hash,b,dis/(2*b),-100)


bf_train = bf_ts.only_store_neibour(compressed_data)  # the result it a list and hard to convert to np array

print(len(bf_train[0]))

# _ = plt.hist(compressed_data[0], bins='auto')
# plt.show()
#
# _ = plt.hist(bf_train[0], bins='auto')
# plt.show()

adict= {}
adict['raw'] = compressed_data[0]
adict['bf5'] = bf_train[0]






dis = float(0.5)


# g=(2*b+1)*com_to_len
false_positive= math.pow( 1-math.exp(-float(num_hash*2*b*com_to_len)/length) , num_hash )
print ('lenth:',length,'num_hash:',num_hash,'total distance:',dis,\
       'step dis:',dis/(2*b),'neighbours:',b, 'false_positive: ', false_positive)
bf_ts=BF_TS(length,num_hash,b,dis/(2*b),-100)
bf_train = bf_ts.only_store_neibour(compressed_data)  # the result it a list and hard to convert to np array
adict['bf05'] = bf_train[0]


dis = float(0.05)


# g=(2*b+1)*com_to_len
false_positive= math.pow( 1-math.exp(-float(num_hash*2*b*com_to_len)/length) , num_hash )
print ('lenth:',length,'num_hash:',num_hash,'total distance:',dis,\
       'step dis:',dis/(2*b),'neighbours:',b, 'false_positive: ', false_positive)
bf_ts=BF_TS(length,num_hash,b,dis/(2*b),-100)
bf_train = bf_ts.only_store_neibour(compressed_data)  # the result it a list and hard to convert to np array
adict['bf005'] = bf_train[0]



str = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/yale_one_distribution.mat'

sio.savemat(str,adict)