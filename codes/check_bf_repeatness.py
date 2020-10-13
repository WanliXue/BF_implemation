import numpy as np
from BF_TS import BF_TS
import math



# load data
data_train=np.load('../data/HAR_traindata.npy') #(7352,561)
data_test=np.load('../data/HAR_testdata.npy')  #(2947 , 561)
label_train=np.load('../data/HAR_trainlabel.npy') #(7352,)
label_test=np.load('../data/HAR_testlabel.npy') #(2947,)

# -----COMPRESS------
compress_matrix = np.load('../data/HAR_compress.npy')
print 'Compress'
compressed_data = np.matmul(compress_matrix,data_train.T).T
# print compressed_data.shape  #(7352,50)
compressed_test_data=np.matmul(compress_matrix,data_test.T).T
# print compressed_test_data.shape #(2947,50)
print 'Compress done'
# ----------



# -------ML on compressed and BF data--------
length = 10000
b = 10
num_hash = 60
dis = float(5)

g=(2*b+2)*50
false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )
print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive

# -----BF-----
# generate the npy with the bf and data
bf_ts=BF_TS(length,num_hash,b,dis/(2*b),dis)

print 'BF start'
bf_train = bf_ts.convert_set_to_bf(compressed_data)  # the result it a list and hard to convert to np array
# bf_test = bf_ts.convert_set_to_bf(compressed_test_data)
print 'BF filter done'

#

har_train =bf_ts.convert_bitarray_to_train_data(bf_train,len(bf_train),length)
# har_test =bf_ts.convert_bitarray_to_train_data(bf_test,len(bf_test),length)

np.save('../data/HAR_train_bf_for_repeatnesscheckfp94',har_train)

# har_train = np.load('../data/HAR_train_bf_for_repeatnesscheck.npy')

print len(har_train)
print har_train[0]

unique_data = [list(x) for x in set(tuple(x) for x in har_train)]
print len(unique_data)