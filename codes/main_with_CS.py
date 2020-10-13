import os
import pandas
import numpy as np
import math
import matplotlib as mpl
mpl.use('TkAgg')  #for mac using matplotlib
import matplotlib.pyplot as plt

from BF_TS import BF_TS


segment_length=60

name='Andrew'

def read_dataset(name):
    database = '..' + os.sep + 'data' + os.sep + 'data_%s.txt' % name
    series = pandas.read_csv(database, header=-1, delimiter='\t')
    hb_array = np.array(series[:][1], dtype=pandas.Series)
    hb_array.astype(int)
    return hb_array

# return the data in datasegment with length (segment_length) e.g.,60
# hb_array is pdarray
def save_segment_data(hb_array,segment_length):
    rec_dic = {}
    total_length = len(hb_array)
    assert total_length >= segment_length, ("too less", total_length)
    # print total_length #32849

    amount_of_rec = total_length / segment_length
    # print amount_of_rec  #547

    data_record = hb_array[:total_length]
    # print len(data_record)
    for i in range(amount_of_rec):
        rec_dic[i]=data_record[i*segment_length:(i*segment_length)+segment_length]
        rec_dic[i].astype(int)
    return rec_dic



def reshape_data(hb_array,segment_length):

    total_length = len(hb_array)
    amount_of_rec = total_length / segment_length
    # print amount_of_rec  #547
    data_record = hb_array[:amount_of_rec*segment_length]
    reshaped = np.array((amount_of_rec, segment_length),dtype=np.float)
    reshaped= np.reshape(data_record,(amount_of_rec,segment_length))
    return reshaped.astype(np.float)


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
    print 'nei_list lenth: ',len(set(nei_list))
    num_hash = int(math.ceil(float(length / len(set(nei_list))) * np.log(2)))


    return num_hash,dis



# -------------------------

# read all data
hb_array=read_dataset(name)

# print type(hb_array[0])

# save all data record in dictonary
rec_dic=save_segment_data(hb_array,segment_length)

# print rec_dic[0]


# reshape to data with (547,60), hb_data[0,:] is the 1st 60 heart beat
hb_data=reshape_data(hb_array,segment_length)

# print hb_data.shape
# print hb_data[0,:] #1st 60 heart beat  0-546
# # print hb_data[546,:]   # the 547-th data segment with index 546

# print np.array_equal(hb_data[0,:] , rec_dic[0])  # return true means to array are same
# # check-check two data are exact same
# for i in range(547):
#     if not np.array_equal(hb_data[i,:] , rec_dic[i]):
#         print 'false'


# compressed_hb=np.array((547,6),dtype=np.float)

##-----compress------
random_m=np.load('../data/random_matrix.npy')  #shape (5,60)

compressed=np.matmul(random_m,hb_data.T)  # (5,60)   * (547,60).T
compress_hb=compressed.T
print compress_hb.shape  #(547,5)  compress_hb[0,:] is 1st

print compress_hb[0,:]

#------ BF CS------------
length=10000     # bf length
b=8       # paramenter for neibours amount
method=0  #use what to calcualte the actual distance 0:uclidean 1:pearson 2:spearman

# all_range=np.reshape(compress_hb,2735)
# num_hash,dis=cal_BF_parrameter(all_range,b,length) #b=3 length=10000
# print num_hash,' ---' ,dis  #3, 10

num_hash=20
dis=50

bf_ts=BF_TS(length,num_hash,b,dis/(2*b),dis)


pair_number=100
# generate 100 pairs in 547elements
pair_array = bf_ts.generate_random_pairs(547,pair_number)


actual_distance=[]
compressed_distance=[]
bf_difference=[]

#------------ for  actual distance vs compressed distance----------------
# for i in range(pair_number):
#
#     data1 = hb_data[pair_array[i][0],:]
#     data2 = hb_data[pair_array[i][1],:]
#     actual_distance.append('%.3f' % (bf_ts.cal_abs_diff(data1, data2, method)))
#
#     cs_data1=compress_hb[pair_array[i][0],:]
#     cs_data2=compress_hb[pair_array[i][1],:]
#     compressed_distance.append('%.3f' % (bf_ts.cal_abs_diff(cs_data1, cs_data2, method)))
#
# print len(actual_distance)
# print len(compressed_distance)
#
# actual_distance=np.array(actual_distance).astype(np.float)
# compressed_distance=np.array(compressed_distance).astype(np.float)
#
# # normalize the result to 0-1 and decimal 3
# result_actual=np.around(  (actual_distance-np.min(actual_distance))/np.ptp(actual_distance), decimals=4  )
# result_compressed=np.around(   (compressed_distance-np.min(compressed_distance))/np.ptp(compressed_distance), decimals=4 )
#
# print result_actual
#
# # -------plot----------
# x=np.arange(0.0,1.0,0.1)
#
# # plt.plot(x,result_absolu,x,result_bf)
#
# plt.plot(1-result_actual,1-result_compressed,'o')
# plt.plot(x,x)
# plt.ylabel('actual_distance')
# plt.xlabel('compressed_distance')
# plt.show()
# #-------




# ---------for actual distance vs bf_cs distance-----
# a=bf_ts.add_neibour(20.680,'a')
# b=bf_ts.add_neibour(21,'a')
# c=bf_ts.add_neibour(22.68,'b')
# print a
# print b
# print c


print hb_data.shape  #(547,60)
print compress_hb.shape  #(547,60)

for i in range(pair_number):

    data1 = hb_data[pair_array[i][0],:]
    data2 = hb_data[pair_array[i][1],:]
    actual_distance.append('%.3f' % (bf_ts.cal_abs_diff(data1, data2, method)))

    # bf_CS distance

    cs_data1=compress_hb[pair_array[i][0],:]
    cs_data2=compress_hb[pair_array[i][1],:]

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

# -------plot-------
x=np.arange(0.0,1.0,0.1)

# plt.plot(x,result_absolu,x,result_bf)

plt.plot(1-result_absolu,result_bf,'o')
plt.plot(x,x)
plt.ylabel('actual_distance')
plt.xlabel('bl_distance')
plt.title('len = %s, b= %s, distance= %s, num_hash=%s'%(length,b,dis,num_hash))
plt.show()
#

# # ---------



# ----find the closed m----
# given a data record
#
# m=5 # top - 5
# n=0  # random one
# da = hb_data[n,:]
# da_bl = compress_hb[n,:]
#
# csbf_input = bf_ts.convert_to_cs_bf_input(da_bl)
# csbf_bfset = bf_ts.__set_to_bloom_filter__(csbf_input)
#
#
# di_actual=np.empty((547),dtype=np.float)
# bl_distance=np.empty((547),dtype=np.float)
#
#
# for i in range(547):
#
#     di_actual=np.append (di_actual, bf_ts.cal_abs_diff(da, hb_data[i,:], method))
#
#     csbf_input_t = bf_ts.convert_to_cs_bf_input(compress_hb[i,:])
#     csbf_bfset_t = bf_ts.__set_to_bloom_filter__(csbf_input_t)
#
#     bl_distance=np.append(bl_distance, bf_ts.cal_diff_two_bf   (csbf_bfset_t,csbf_bfset))
#
# print di_actual.argsort()[:m]
# print bl_distance.argsort()[:m]

# ---------------------------------------