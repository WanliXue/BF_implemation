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

# ----- end of parametter
# ---------------------------------

# read all data
hb_array=read_dataset(name)

# print type(hb_array[0])

# save all data record in dictonary
rec_dic=save_segment_data(hb_array,segment_length)


# check - check
# for key, value in rec_dic.iteritems():
#     print key, value.dtype

# a=[74, 75, 75, 74, 74, 74, 73 ,74, 74 ,75, 75, 92, 89, 82 ,77, 74 ,74 ,73 ,73 ,72 ,70, 69 ,70 ,70 , \
#      71 ,72 ,74, 74 ,71, 70, 70, 71, 72 ,74, 74, 73, 73 ,73 ,73, 73, 73 ,73 ,72, 71 ,70, 70 ,72, 71 , \
#      71 ,71, 71 ,69 ,63, 63, 61, 61 ,62, 62 ,66, 64]
#
# print len(a)

# #check-check if all value are Int
# for key,value in rec_dic.iteritems():
#     print key
#     for i in range(60):
#         if isinstance(value[i], int):
#             print "INT",i
#         else:
#             print "NOT INT"




length=10000     # bf length
b=3         # paramenter for neibours amount
method=0  #use what to calcualte the actual distance 0:uclidean 1:pearson 2:spearman


num_hash,dis=cal_BF_parrameter(hb_array,b,length) #b=3 length=2000
print 'mum_hash is: ',num_hash
print 'maximum distance: ', dis

num_hash=20
dis=100

# initialate the Bloom Filter _ Time Series instance
bf_ts=BF_TS(length,num_hash,b,dis/(2*b),dis)
# print rec_dic[0]

# -------check_check-----
a1= bf_ts.add_neibour(64,0)
a2= bf_ts.add_neibour(65,0)
a3= bf_ts.add_neibour(66,0)
a4= bf_ts.add_neibour(67,0)
print a1,'\n',a2,'\n',a3,'\n',a4


b1=bf_ts.__set_to_bloom_filter__(a1)
b2=bf_ts.__set_to_bloom_filter__(a2)
b3=bf_ts.__set_to_bloom_filter__(a3)
b4=bf_ts.__set_to_bloom_filter__(a4)

print '64 65 bf difference: ', bf_ts.cal_diff_two_bf(b1,b2)
print '64 66 bf difference: ', bf_ts.cal_diff_two_bf(b1,b3)
print '64 66 bf difference: ', bf_ts.cal_diff_two_bf(b1,b4)
print '66 67 bf difference: ', bf_ts.cal_diff_two_bf(b3,b4)
# ------------

# --------check-check
# bf1_input=bf_ts.convert_to_bf_input(rec_dic[2])
# bf1= bf_ts.__set_to_bloom_filter__(bf1_input)
#
#
# bf2_input=bf_ts.convert_to_bf_input(rec_dic[1])
# bf2= bf_ts.__set_to_bloom_filter__(bf2_input)

# print 'absolute diff: ', bf_ts.cal_abs_diff(rec_dic[2],rec_dic[1],0)
# print 'bf difference: ', bf_ts.cal_diff_two_bf(bf1,bf2)
# ---------

# ----check-check-----
bf1_input=bf_ts.convert_to_bf_input([64,65,67])
bf2_input=bf_ts.convert_to_bf_input([64,65,70])
# print len(bf1_input)
bf1=bf_ts.__set_to_bloom_filter__(bf1_input)
bf2=bf_ts.__set_to_bloom_filter__(bf2_input)
print 'bf difference: ', bf_ts.cal_diff_two_bf(bf1,bf2)

# -------------------



pair_number=100
# generate 100 pairs in 547elements
pair_array = bf_ts.generate_random_pairs(547,pair_number)
print len(pair_array)

#evalue the difference of raw and bf_vectors

# raw/absolute difference and differece after bf
raw_difference = []
bf_difference = []

for i in range(pair_number):
    # print pair_array[i]

    data1=rec_dic[pair_array[i][0]]
    data2=rec_dic[pair_array[i][1]]

    # method=0 1 2 3 uclidean spearman..
    # if method =1 get uclidean distance, so the final reulst shoulda be 1-resul
    raw_difference.append('%.3f'%(bf_ts.cal_abs_diff(data1, data2, method) ) )

    bf_difference.append('%.3f'% (  bf_ts.cal_bf_diff(data1, data2 )  ) )



# #check check
# data=np.array(['142.962', '235.900', '140.773', '183.829', '137.408', '221.072', '236.112', '35.749', '218.563', '52.278', '149.369', '237.950']).astype(np.float)
# print 'min', np.min(data)
# result=(data-np.min(data))/np.ptp(data)
# a=np.around(result,decimals=3)
# print len(a)

# print 'lengeth',len(raw_difference)
# print 'absolute diff: ' ,raw_difference
# print 'bf difference: ', bf_difference
#

absolu_dif=np.array(raw_difference).astype(np.float)
bl_dif=np.array(bf_difference).astype(np.float)

# normalize the result to 0-1 and decimal 3
result_absolu=np.around(  (absolu_dif-np.min(absolu_dif))/np.ptp(absolu_dif), decimals=4  )
result_bf=np.around(   (bl_dif-np.min(bl_dif))/np.ptp(bl_dif), decimals=4  )

# print 'absulut diffrerece: ', result_absolu
# print 'bf difference: ', result_bf


# -------plot-------
x=np.arange(0.0,1.0,0.1)

# plt.plot(x,result_absolu,x,result_bf)

plt.plot(1-result_absolu,result_bf,'o')
plt.plot(x,x)
plt.ylabel('actual_distance')
plt.xlabel('bl_distance')
plt.show()
#

