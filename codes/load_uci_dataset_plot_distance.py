import os
import pandas
import numpy as np
import math
import matplotlib as mpl
mpl.use('TkAgg')  #for mac using matplotlib
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 20})

from BF_TS import BF_TS

name='HAR_train'


com_to_len = 50


def save_har_dataset(name):
    data_har=[]
    database = '..' + os.sep + 'data' + os.sep + 'data_%s.txt' % name
    series = pandas.read_csv(database, header=-1, delimiter='\t')
    t_data = np.array(series, dtype=pandas.Series)
    for i in range(len(t_data)):
        data = t_data[i][0].split(' ')
        # data = map(float, data)
        data_har.append(filter(None , data))
        # print len(data_har)
        # print i

    data_har = np.array(data_har,dtype=np.float)
    return data_har

# ---read data and save in npy (if change the compression ratio, need do again)---
# data_har = save_har_dataset(name)
# np.save('HAR_testdata',data_har)

data_har=np.load('../data/HAR_traindata.npy')

print data_har.shape   #(73214, 561)
# print data_har[0]
# print len(data_har[0])  #561
# print type(data_har[0][1])

# -------- SVD the data set get the compression matrix---------
# u,s,vh = np.linalg.svd(data_har.T,full_matrices= True)
# print u.shape
#
# # get the compression matrix
# compress_matrix = u[:,0:com_to_len].T
# print compress_matrix.shape  # (com_to_len , 561)
# np.save('HAR_compress',compress_matrix)
# ----------------------

compress_matrix=np.load('../data/HAR_compress.npy')

f_column = np.matmul(compress_matrix,data_har.T)
# print len(f_column[0])
# print f_column[0]

# print f_column.shape
# print f_column.T[0]  # f = R (50x 561) * X (561 x 7352)
# print len(f_column.T[0])

compress_data = f_column.T
print compress_data.shape  #(7352,50)

# calcualte number of hash function and maximum distance
def cal_BF_parrameter(hb_array,b,length):

    dis = 0.05 * (max(hb_array) - min(hb_array)) #used to be 0.05

    nei_list = []
    uniq_elem_list = set(hb_array)
    for uniq_elem in uniq_elem_list:
        nei_list.append(uniq_elem)
        rem_uniq_elem_val = uniq_elem % (dis / b)  # Convert into values within same interval
        if rem_uniq_elem_val >= 0.5 * (dis / b):
            nei_list.append(uniq_elem + ((dis / b) - rem_uniq_elem_val))
        else:
            nei_list.append(uniq_elem - rem_uniq_elem_val)
    num_hash = int(math.ceil(float(length / len(set(nei_list))) * np.log(2)))


    return num_hash,dis


length=10000    # bf length 30000 10000
b= 10 # paramenter for neibours amount  10
method=0  #use what to calcualte the actual distance 0:uclidean 1:pearson 2:spearman

# num_hash,dis=cal_BF_parrameter(compress_data[0],b,length) #b=3 length=2000
# print 'mum_hash is: ',num_hash #107
# print 'maximum distance: ', dis  #1.11



num_hash=30  #20 30
dis=float(5)  # small distance make more convenge


g=(2*b+1)*com_to_len
false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )
print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive




bf_ts=BF_TS(length,num_hash,b,dis/(2*b),dis)

pair_number=100
# generate 100 pairs in 547elements
pair_array = bf_ts.generate_random_pairs(7352,pair_number)

actual_distance=[]
compressed_distance=[]
bf_difference=[]

# ---------for actual distance vs bf_cs distance-----
# data_train=np.load('../data/HAR_traindata.npy') #(7352,561)
#
# compress_matrix = np.load('../data/HAR_compress.npy')
# print 'Compress'
# compressed_data = np.matmul(compress_matrix,data_train.T).T
#
# print 'BF start'
# bf_train = bf_ts.convert_set_to_bf(compressed_data)  # the result it a list and hard to convert to np array
# print 'BF filter done'
# # # print bf_train[0]
# # # print bf_train[0].to01()
#
# har_train = bf_ts.convert_bitarray_to_train_data(bf_train, len(bf_train), length)
#
# np.save('../data/HAR_train_inputdata',har_train)
#
# np.load('../data/HAR_train_inputdata')
#

# for i in range(pair_number):
#     list1=pair_array[i][0]
#     list2=pair_array[i][0]
#
# bf_dist_test1=har_train[list1]
# bf_dist_test2=har_train[list2]

for i in range(pair_number):

    data1 = data_har[pair_array[i][0],:]
    data2 = data_har[pair_array[i][1],:]
    actual_distance.append('%.3f' % (bf_ts.cal_abs_diff(data1, data2, method)))

    # bf_CS distance

    cs_data1=compress_data[pair_array[i][0],:]
    cs_data2=compress_data[pair_array[i][1],:]

    cs_data1=np.around((cs_data1), decimals=2)
    cs_data2 = np.around((cs_data2), decimals=2)

    csbf_input1 = bf_ts.convert_to_cs_bf_input(cs_data1)
    csbf_input2 = bf_ts.convert_to_cs_bf_input(cs_data2)

    csbf_bfset1 = bf_ts.__set_to_bloom_filter__(csbf_input1)
    csbf_bfset2 = bf_ts.__set_to_bloom_filter__(csbf_input2)

    # bf_difference.append('%.3f' %    (bf_ts.cal_diff_two_bf   (csbf_bfset1,csbf_bfset2)  )  )

    csbf_bfset1 = bf_ts.convert_one_bitarray_to_train_data(csbf_bfset1, len(csbf_bfset1), length)
    csbf_bfset2 = bf_ts.convert_one_bitarray_to_train_data(csbf_bfset2, len(csbf_bfset2), length)
    bf_difference.append('%.3f' % (bf_ts.cal_abs_diff(csbf_bfset1, csbf_bfset2, 4)))




absolu_dif = np.array(actual_distance).astype(np.float)
bl_dif = np.array(bf_difference).astype(np.float)


# normalize the result to 0-1 and decimal 3
result_absolu=np.around(  (absolu_dif-np.min(absolu_dif))/np.ptp(absolu_dif), decimals=4  )
result_bf=np.around(   (bl_dif-np.min(bl_dif))/np.ptp(bl_dif), decimals=4  )



# How_far_away= bf_ts.cal_abs_diff(result_absolu,result_bf,0)

rms=np.sqrt(np.mean((result_absolu-result_bf)**2))
print 'rms: ',rms
rms=np.around(   rms, decimals=2 )


# -------plot-------
x=np.arange(0.0,1.0,0.1)

# plt.plot(x,result_absolu,x,result_bf)

# plt.plot(result_bf,result_absolu,'o')
plt.plot(result_absolu,result_bf,'o')
plt.plot(x,x)
plt.ylabel('Actual distance similarity')
plt.xlabel('Distance similarity in Bloom filter space')
# plt.title('len = %s, b= %s, distance= %s, num_hash=%s, fp=%s, rms=%s'%(length,b,dis,num_hash,false_positive,rms))
plt.title('rms=%s'%(rms))

plt.show()
#

# # ---------
