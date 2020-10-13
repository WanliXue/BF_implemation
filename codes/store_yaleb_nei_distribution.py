from BF_TS_2digit_neigdis import BF_TS
import math
import numpy as np
import time
from datetime import datetime
import scipy.io as sio


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



compressed_data = train_data


length = 10000
b=5
num_hash = 1
dis = float(0.05)
com_to_len = 561

# g=(2*b+1)*com_to_len
false_positive= math.pow( 1-math.exp(-float(num_hash*2*b*com_to_len)/length) , num_hash )
print ('lenth:',length,'num_hash:',num_hash,'total distance:',dis,\
       'step dis:',dis/(2*b),'neighbours:',b, 'false_positive: ', false_positive)
bf_ts=BF_TS(length,num_hash,b,dis/(2*b),-100)


bf_train = bf_ts.only_store_neibour(compressed_data)  # the result it a list and hard to convert to np array
print(len(bf_train[0]))

filename = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/yaleb_dis005.npy'
np.save(filename, bf_train)
# print('using time, ', time.time() - timenow)