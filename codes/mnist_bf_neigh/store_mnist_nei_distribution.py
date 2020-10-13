from codes.BF_TS_2digit_neigdis import BF_TS
import math
import time
from datetime import datetime
import numpy as np
import scipy.io



data_test=np.load('/Users/wanli/Desktop/sanjay_mnist/data/mnist_test_inputdata_full.npy') #(7352,561)

print(data_test.shape)


# compress_matrix=np.load('/Users/wanli/Desktop/sanjay_mnist/data/MNIST_compress.npy')
# print(compress_matrix.shape)
# #-----------------------
# ## compress
#
# print ('Compress')
# #(7352,60)
# compressed_test_data=np.matmul(compress_matrix,data_test).T
# # print compressed_test_data.shape #(2947,60)
# print ('Compress done')
#
# # np.save('./data/compressed_train_data_full',compressed_data)
# # np.save('./data/compressed_test_data_full',compressed_test_data)
#
#
#
#
# # #save to matlab file
# # adict= {}
# # adict['cha1'] = compressed_test_data
# # str = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/mnist_SVD.mat'
# #
# # scipy.io.savemat(str,adict)
# ##--------------------
compressed_test_data = data_test


length = 10000
b=10
num_hash = 2
dis = float(0.05)
com_to_len = 784  # 784 or 60

# g=(2*b+1)*com_to_len
false_positive= math.pow( 1-math.exp(-float(num_hash*2*b*com_to_len)/length) , num_hash )
print ('lenth:',length,'num_hash:',num_hash,'total distance:',dis,\
       'step dis:',dis/(2*b),'neighbours:',b, 'false_positive: ', false_positive)
bf_ts=BF_TS(length,num_hash,b,dis/(2*b),-100)


bf_train = bf_ts.only_store_neibour(compressed_test_data)  # the result it a list and hard to convert to np array
print(len(bf_train[0]))

# filename = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/mnist_svd_dis0.25.npy'
filename = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/mnist_dis00025.npy'
np.save(filename, bf_train)
