import numpy as np
# import os
# print(os.getcwd())
import math
import time
from BF_TS import BF_TS
import datetime as dt


def conver_data_to_bf(data):
    com_to_len = data.shape[1]
    # print(data.shape)
    #
    compressed_data = data[:, :, 0]
    print (compressed_data.shape)
    #
    compressed_data2 = data[:, :, 1]
    compressed_data3 = data[:, :, 2]
    #
    #
    # ------------Put into  compressed and BF data---------
    length = 16000
    b = 5
    num_hash = 2
    dis = float(5)

    g = (2 * b + 1) * com_to_len
    false_positive = math.pow(1 - math.exp(-(num_hash * g) / length), num_hash)

    # print ('lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive)

    false_positive = math.pow(1 - math.exp(-(float)(num_hash * g) / length), num_hash)
    print ('lenth:', length, 'num_hash:', num_hash, 'false_positive: ', false_positive)

    ## # generate the npy with the bf and data
    bf_ts = BF_TS(length, num_hash, b, 0.5, 1000)
    ##

    # ---------------------
    print ('BF filter')
    # print 'start'
    start_time = time.time()
    bf_train = bf_ts.convert_set_to_bf(compressed_data)  # the result it a list and hard to convert to np array
    bf_train2 = bf_ts.convert_set_to_bf(compressed_data2)
    bf_train3 = bf_ts.convert_set_to_bf(compressed_data3)
    print ('BF filter done')

    cifar_batch = bf_ts.convert_bitarray_to_train_data(bf_train, len(bf_train), length)
    cifar_batch2 = bf_ts.convert_bitarray_to_train_data(bf_train, len(bf_train2), length)
    cifar_batch3 = bf_ts.convert_bitarray_to_train_data(bf_train, len(bf_train3), length)

    # # put into the one file saved in npy/ stack 3 dimension together
    # cifar_batch=np.array([cifar_batch,cifar_batch2,cifar_batch3])
    # print(cifar_batch.shape)
    # # transpose (3,9000,10000) to (9000,10000,3))
    # cifar_bfed = cifar_batch.transpose(1,2,0)
    # print(cifar_bfed.shape)
    # # np.save('../data/cifar_test_bfed.npy',cifar_bfed)

    print('bf done using time: {} mins'.format((time.time() - start_time) / 60))

    cifar_bfed = np.stack([cifar_batch, cifar_batch2, cifar_batch3], axis=2)
    return cifar_bfed

    # ---------------------


# ----------------
# data = np.load("../data/cifar-test-batch.npy")
# print(data.shape)

# x1=np.load("../data/batch_1.npy")
# #
# x2=np.load("../data/batch_2.npy")
# x3=np.load("../data/batch_3.npy")
# x4=np.load("../data/batch_4.npy")
# x5=np.load("../data/batch_5.npy")

# print(x5.shape)
# print(x1[0].shape)
# print(x1[101][30][31][0])

# y1=x1.reshape(9000,1024,3)
# print(y1.shape)
# print(y1[101][991][0])



# y3=x3.reshape(9000,1024,3)
# y2=x2.reshape(9000,1024,3)
# y4=x4.reshape(9000,1024,3)
# y5=x5.reshape(9000,1024,3)

# print(y1[0][991][0])
# print(y2[0][991][0])
#
# import scipy.io as si

# si.savemat('cifar_batch5.mat',dict(x=y5))

# -------------------------

path = '/Users/wanli/Dropbox/ppml_code_with_dataset/CIFAR_mnist/cifar_batch'


for batch_id in range(1,6):
    filename = path + str(batch_id) + '_random300.npy'
    print(filename)
    data = np.load(filename)  # (9000,300,3)
    bfed = conver_data_to_bf(data)
    save_path = '../data/cifar_batch' + str(batch_id) + '_bfed_random300.npy'
    np.save(save_path, bfed)
#
#
# ------------------

filename = '/Users/wanli/Dropbox/ppml_code_with_dataset/CIFAR_mnist/cifar_test_random300.npy'
print(filename)
data = np.load(filename)  # (9000,300,3)
bfed = conver_data_to_bf(data)
save_path = '../data/cifar_test_bfed_random300.npy'
np.save(save_path, bfed)

print(dt.datetime.now())