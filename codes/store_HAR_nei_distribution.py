from BF_TS_2digit_neigdis import BF_TS
import math
import numpy as np
import time
from datetime import datetime
data_train=np.load('../data/HAR_traindata.npy') #(7352,561)



# compress_matrix = np.load('../data/HAR_compress.npy')
# print 'Compress'
# compressed_data = np.matmul(compress_matrix,data_train.T).T
# print compressed_data.shape  #(7352,50)
# print 'Compress done'


compressed_data = data_train


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

filename = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/har_dis005.npy'
np.save(filename, bf_train)
# print('using time, ', time.time() - timenow)