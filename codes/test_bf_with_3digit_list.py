from BF_TS_2digit_neigdis import BF_TS
import math
import numpy as np
import time
from datetime import datetime





length = 10000
b=5
num_hash = 1
dis = float(0.05)
com_to_len = 1024

# g=(2*b+1)*com_to_len
false_positive= math.pow( 1-math.exp(-float(num_hash*2*b*com_to_len)/length) , num_hash )
print ('lenth:',length,'num_hash:',num_hash,'total distance:',dis,\
       'step dis:',dis/(2*b),'neighbours:',b, 'false_positive: ', false_positive)

bf_ts=BF_TS(length,num_hash,b,dis/(2*b),-100)

batch = 1
print('batch ', batch)

filename = '/Users/wanli/Desktop/tmp/aruna/raw_cifar_preprocess/preprocess_batch_'+str(batch)+'.npy'
cifar_data = np.load(filename)
print(cifar_data.shape)


testdata = cifar_data[0:2,:,0]
print(testdata.shape)

bf_train = bf_ts.convert_set_to_bf(testdata)  # the result it a list and hard to convert to np array
output1 = bf_ts.convert_bitarray_to_train_data(bf_train, len(bf_train), length)

print(output1)
print(output1.shape)