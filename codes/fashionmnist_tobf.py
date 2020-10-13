import numpy as np
# import os
# print(os.getcwd())
import math
import time
from BF_TS import BF_TS


def conver_data_to_bf(data):
    com_to_len = data.shape[1]
    # print(data.shape)
    #
    compressed_data = data[:, :]

    #
    # ------------Put into  compressed and BF data---------
    length = 10000
    b = 5
    num_hash = 10
    dis = float(5)

    g = (2 * b + 1) * com_to_len
    false_positive = math.pow(1 - math.exp(-(num_hash * g) / length), num_hash)

    # print ('lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive)

    false_positive = math.pow(1 - math.exp(-(float)(num_hash * g) / length), num_hash)
    print ('lenth:', length, 'num_hash:', num_hash, 'false_positive: ', false_positive)

    ## # generate the npy with the bf and data
    bf_ts = BF_TS(length, num_hash, b, dis / (2 * b), dis)
    ##

    # ---------------------
    print ('BF filter')
    # print 'start'
    start_time = time.time()
    bf_train = bf_ts.convert_set_to_bf(compressed_data)  # the result it a list and hard to convert to np array

    print ('BF filter done')

    cifar_batch = bf_ts.convert_bitarray_to_train_data(bf_train, len(bf_train), length)


    print('bf done using time: {} mins'.format((time.time() - start_time) / 60))

    # cifar_bfed = np.stack([cifar_batch, cifar_batch2, cifar_batch3], axis=2)
    return cifar_batch

    # ---------------------
train_path = '/Users/wanli/Dropbox/ppml_code_with_dataset/CIFAR_mnist/Fashion_train_random60_full.npy'

test_path = '/Users/wanli/Dropbox/ppml_code_with_dataset/CIFAR_mnist/Fashion_test_random60_full.npy'

data = np.load(train_path)  # (9000,300,3)
bfed = conver_data_to_bf(data)
save_path = '../data/fashion_bfed_train_random60.npy'
np.save(save_path, bfed)

data_test = np.load(test_path)  # (9000,300,3)
bfed_test = conver_data_to_bf(data_test)
save_path = '../data/fashion_bfed_test_random60.npy'
np.save(save_path, bfed_test)