import numpy as np
import scipy.io as sio
import math
import time
from datetime import datetime
from BF_TS_2digit import BF_TS
import datetime as dt
import matplotlib.pyplot as plt
import pickle

# raw_data = np.load('/Users/wanli/Desktop/bf_deep/saved_cifar_data/cifar_batch1.npy')

# raw_data = np.load('/Users/wanli/Desktop/bf_deep/saved_cifar_data/batch1_0.npy')
# print(raw_data.shape)

# raw_data = '/Users/wanli/Desktop/bf_deep/saved_cifar_data/cifar_batch1.mat'
# cifar_data = sio.loadmat(raw_data)
# cifar_data = cifar_data['x']
# print(cifar_data.shape)
# # print(cifar_data['x'][0])


length = 50000
b=5
num_hash = 4
dis = float(0.03)
com_to_len = 1024

# g=(2*b+1)*com_to_len
false_positive= math.pow( 1-math.exp(-float(num_hash*2*b*com_to_len)/length) , num_hash )
print ('lenth:',length,'num_hash:',num_hash,'total distance:',dis,\
       'step dis:',dis/(2*b),'neighbours:',b, 'false_positive: ', false_positive)


bf_ts=BF_TS(length,num_hash,b,dis/(2*b),-100)

timenow = time.time()
print(datetime.now())


# expected width distance = 0.003

batch = 1
print('batch ', batch)

filename = '/Users/wanli/Desktop/tmp/aruna/raw_cifar_preprocess/preprocess_batch_'+str(batch)+'.npy'
cifar_data = np.load(filename)
print(cifar_data.shape)

for i in range(3):
    print(i)
    chan = cifar_data[:,:,i]
    print(chan.shape)

    bf_train = bf_ts.convert_set_to_bf(chan)  # the result it a list and hard to convert to np array
    output1 =bf_ts.convert_bitarray_to_train_data(bf_train,len(bf_train),length)
    filename = '/Users/wanli/Desktop/bf_deep/saved_cifar_data/bfed_batch'+str(batch)+'_'+ str(i) + '.npy'
    np.save(filename,output1)
    print('using time, ', time.time() - timenow)

# test
# chan = cifar_data[0:2,:,0]
# bf_train = bf_ts.convert_set_to_bf(chan)  # the result it a list and hard to convert to np array
# output1 =bf_ts.convert_bitarray_to_train_data(bf_train,len(bf_train),length)
# filename = '/Users/wanli/Desktop/bf_deep/saved_cifar_data/batch1_' + str(0) + '.npy'
# np.save(filename,output1)

