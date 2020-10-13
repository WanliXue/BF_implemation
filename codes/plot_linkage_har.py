import os
import pandas
import numpy as np
import math
import matplotlib as mpl
import scipy.io
mpl.use('TkAgg')  #for mac using matplotlib
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.size': 20})

from BF_TS import BF_TS



#
# data=scipy.io.loadmat('../data/csi.mat')
#
# data_train=data['train_set']
# data_test = data['test_set']
# label_train = data['train_label']
# label_test = data['test_label']
#
# # data_train = np.delete(data_train,0,1)
# # data_test = np.delete(data_test,0,1)
#
# print data_train.shape
# print data_test.shape
# compressed_data = data_train
# compressed_test_data = data_test


# har data
data_har=np.load('../data/HAR_traindata.npy')


#har compressed data
compress_matrix=np.load('../data/HAR_compress.npy')
f_column = np.matmul(compress_matrix,data_har.T)
compress_data = f_column.T




com_to_len = 50
# ---------------BF'ed data ML -----------
length = 10000
b =10
num_hash = 30
dis = float(5)

g=(2*b+2)*com_to_len
false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )

print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive

# # generate the npy with the bf and data
bf_ts=BF_TS(length,num_hash,b,dis/(2*b),dis)

# --------------

pair_number=100
# generate 100 pairs in 7352elements
pair_array = bf_ts.generate_random_pairs(7352,pair_number)

actual_distance=[]
compressed_distance=[]
bf_difference=[]
bfdp_difference=[]
method=0  #use what to calcualte the actual distance 0:uclidean 1:pearson 2:spearman



#
# #------------ for  actual distance vs compressed distance----------------
# for i in range(pair_number):
#
#     data1 = data_har[pair_array[i][0],:]
#     data2 = data_har[pair_array[i][1],:]
#     actual_distance.append('%.3f' % (bf_ts.cal_abs_diff(data1, data2, method)))
#
#     cs_data1=compress_data[pair_array[i][0],:]
#     cs_data2=compress_data[pair_array[i][1],:]
#     compressed_distance.append('%.3f' % (bf_ts.cal_abs_diff(cs_data1, cs_data2, method)))
#
# print len(actual_distance)
# print len(compressed_distance)
#
# actual_distance=np.array(actual_distance).astype(np.float)
# compressed_distance=np.array(compressed_distance).astype(np.float)
#
#
# # normalize the result to 0-1 and decimal 4
# result_actual=np.around(  (actual_distance-np.min(actual_distance))/np.ptp(actual_distance), decimals=4  )
# result_compressed=np.around(   (compressed_distance-np.min(compressed_distance))/np.ptp(compressed_distance), decimals=4 )
#
# # print result_actual
#
# # -------plot----------
# x=np.arange(0.0,1.0,0.1)
#
# # plt.plot(x,result_absolu,x,result_bf)
#
#
# plt.plot(1-result_actual,1-result_compressed,'o')
# plt.plot(x,x)
# plt.ylabel('Actual distance similarity')
# plt.xlabel('Compressed distance similarity')
# plt.show()
# # -------
# # ---------------------------------------


#
#------------ for  actual distance vs bf distance----------------
for i in range(pair_number):

    data1 = data_har[pair_array[i][0],:]
    data2 = data_har[pair_array[i][1],:]
    actual_distance.append('%.3f' % (bf_ts.cal_abs_diff(data1, data2, method)))

    # bf_CS distance

    cs_data1=compress_data[pair_array[i][0],:]
    cs_data2=compress_data[pair_array[i][1],:]

    csbf_input1 = bf_ts.convert_to_cs_bf_input(cs_data1)
    csbf_input2 = bf_ts.convert_to_cs_bf_input(cs_data2)

    csbf_bfset1 = bf_ts.__set_to_bloom_filter__(csbf_input1)
    csbf_bfset2 = bf_ts.__set_to_bloom_filter__(csbf_input2)

    bf_difference.append('%.3f' %    (bf_ts.cal_diff_two_bf   (csbf_bfset1,csbf_bfset2)  )  )



absolu_dif = np.array(actual_distance).astype(np.float)
bl_dif = np.array(bf_difference).astype(np.float)


# normalize the result to 0-1 and decimal 3
result_absolu=np.around(  (absolu_dif-np.min(absolu_dif))/np.ptp(absolu_dif), decimals=4  )
result_bf=np.around(   (bl_dif-np.min(bl_dif))/np.ptp(bl_dif), decimals=4  )

How_far_away= bf_ts.cal_abs_diff(result_absolu,result_bf,0)

# -------plot-------
x=np.arange(0.0,1.0,0.1)

# plt.plot(x,result_absolu,x,result_bf)

plt.plot(result_absolu,result_bf,'o')
plt.plot(x,x)
plt.ylabel('Actual distance similarity')
plt.xlabel('Bloom filter based distance similarity')
# plt.title('len = %s, b= %s, distance= %s, num_hash=%s, fp=%s, howfar=%s'%(length,b,dis,num_hash,false_positive,How_far_away))
# plt.title('len = %s, b= %s, distance= %s, num_hash=%s, fp=%s, howfar=%s'%(length,b,dis,num_hash,false_positive,How_far_away))
plt.show()
#

# ---------
#
#
#
# dataA=[]
# dataB=[]
# for i in pair_array:
#     dataA.append(i[0])
#     dataB.append(i[1])
#
#
# for i in range(pair_number):
#
#     data1 = data_har[pair_array[i][0],:]
#     data2 = data_har[pair_array[i][1],:]
#     actual_distance.append('%.3f' % (bf_ts.cal_abs_diff(data1, data2, method)))
#


# # -------- for actual distance vs bf_dp_distance---------
#
#
#
# # bf_dp distance calcu
# #
#
# cs_data1 = compress_data[dataA]
# cs_data2 = compress_data[dataB]
#
#
# bf_train1 = bf_ts.convert_set_to_bf(cs_data1)
# bf_train2 = bf_ts.convert_set_to_bf(cs_data2)
#
#
# har_train1 =bf_ts.convert_bitarray_to_train_data(bf_train1,len(bf_train1),length)
# har_train2 =bf_ts.convert_bitarray_to_train_data(bf_train2,len(bf_train2),length)
#
# loc  = 0.  # location = mean
# # scale =1.  # location = mean scale=decay
# epsilon = 1
# sensity =1.0
# print 'epsilon: ',epsilon
# scale = sensity/epsilon
# print 'scale: ', scale
# p=0.2  #noise possibility
#
#
# print 'epsilon :',epsilon,', p-value, ', p
#
#
# har_train1=bf_ts.adding_lp_noise_to_bf_data(har_train1,epsilon,length , p)
# har_train2=bf_ts.adding_lp_noise_to_bf_data(har_train2,epsilon,length , p)
#
# for i in range(100):
#
#     data1 = har_train1[i]
#     data2 = har_train2[i]
#     bfdp_difference.append('%.3f' % (bf_ts.cal_abs_diff(data1, data2, method)))
#
#
#
#
# absolu_dif = np.array(actual_distance).astype(np.float)
# bldp_dif = np.array(bfdp_difference).astype(np.float)
#
# # normalize the result to 0-1 and decimal 3
# result_absolu=np.around(  (absolu_dif-np.min(absolu_dif))/np.ptp(absolu_dif), decimals=4  )
# result_bfdp=np.around(   (bldp_dif-np.min(bldp_dif))/np.ptp(bldp_dif), decimals=4  )
#
# How_far_away= bf_ts.cal_abs_diff(result_absolu,result_bfdp,0)
#
# # -------plot-------
# x=np.arange(0.0,1.0,0.1)
#
# # plt.plot(x,result_absolu,x,result_bf)
#
# plt.plot(1-result_absolu,result_bfdp,'o')
# plt.plot(x,x)
# plt.ylabel('Actual distance similarity')
# plt.xlabel('bldp_distance')
# plt.title('len = %s, b= %s, distance= %s, num_hash=%s, fp=%s, howfar=%s'%(length,b,dis,num_hash,false_positive,How_far_away))
# plt.show()
# #
#
# # ---------


