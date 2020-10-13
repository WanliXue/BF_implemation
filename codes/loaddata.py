import os
import pandas
import numpy as np
import math
import hashlib

from scipy import spatial
from scipy import stats
from bitarray import bitarray


name='Andrew'
database = '..'+os.sep+'data'+os.sep+'data_%s.txt' % name


##load time series data
series = pandas.read_csv(database,header=-1,delimiter='\t')
# print (series.head())


i=60
# df2 = series.head(i)
# print df2

# print series[1][1]

# series[1][1]=74

# #get top i as one record
# hb_data=series[:i][1]  #only get first i data
# # hb_data=series[:][1]  # get all
# print hb_data.size
# hb_data1=series[i:i+i][1]
# # print hb_data1.size
#
# hb_data2=series[i+i:3*i][1]
# print hb_data2.size

# print stats.pearsonr(hb_data,hb_data1)
# print stats.pearsonr(hb_data,series[2*i:3*i][1])

# X = np.c_[hb_data, hb_data1]
# X = [hb_data, hb_data1]
# X=np.array([hb_data,hb_data1]).transpose()
# print type(X), X.shape

# val1,2 are normal time series data (pdarray)
def cal_abs_diff(val1,val2,method):
    if method==0: #euclidean distance
        d=spatial.distance.euclidean(val1,val2)
        return d
    if method==3: #pairwise distance bewteen ob in n-dimension
        # X=np.c_[val1, val2]
        X = np.array([val1, val2])
        y=spatial.distance.pdist(X.transpose(),'seuclidean')
        d=spatial.distance.squareform(y)
        # d=spatial.distance.seuclidean(val1,val2)
        return d
    if method==1: #     pearson corelation (-1,+1)
        [r,_]=stats.pearsonr(val1,val2)
        return r
    if method==2: #spearman  (-1,+1)
        [r,_]=stats.spearmanr(val1,val2)
        return r

# [r,_]= stats.pearsonr([1,2,3,4,5],[1,2,2,4,5])
# print 'r value is ',r

# print cal_abs_diff(hb_data,hb_data1,0)
# print cal_abs_diff(hb_data,hb_data1,1)
# print cal_abs_diff(hb_data,hb_data1,2)
#
# print cal_abs_diff(hb_data,hb_data2,0)
# print cal_abs_diff(hb_data,hb_data2,1)
# print cal_abs_diff(hb_data,hb_data2,2)

# print cal_abs_diff(hb_data,hb_data1,3)


# #---------------




# print hb_data
# print hb_data1
# print series[2*i:3*i][1]

# print series[:i][1]
# print hb_data


hb_array=np.array(series[:][1],dtype=pandas.Series)


hb_data=hb_array[:i]
print hb_data

hb_data1=hb_array[i:i+i]

print 'hb_array shape ',hb_array.shape

b=3 #5 no of interger
d = 0.05*(max(hb_array) - min(hb_array))
length=2000

# print 'maximum distance ',d



def generate_nei_list(int_array):
    nei_list=[]
    uniq_elem_list = set(int_array)
    for uniq_elem in uniq_elem_list:
        nei_list.append(uniq_elem)
        rem_uniq_elem_val = uniq_elem % (d / b)  # Convert into values within same interval
        if rem_uniq_elem_val >= 0.5 * (d / b):
            nei_list.append(uniq_elem + ((d / b) - rem_uniq_elem_val))
        else:
            nei_list.append(uniq_elem - rem_uniq_elem_val)
    return nei_list

nei_list=generate_nei_list(hb_array)


num_hash =int(math.ceil(float(length/len(set(nei_list))) * np.log(2)))
# num_hash = round(float(length/len(set(nei_list))) * np.log(2))

def show_parameter():
    # print 'neilist is: ',generate_nei_list(hb_array)
    # print 'how many in the neighbour ',len(generate_nei_list(hb_array))
    # print nei_list
    print 'no. of hash ',num_hash

show_parameter()

h1 = hashlib.sha1
h2 = hashlib.md5

step_size=d/(2*b)

# val1 1 data instance
def add_neibour(val1, sequence):
    """Calculate absolute difference similarity between two
       numerical values encoded into Bloom filters.
    """

    # Number of intervals and their sizes (step) to consider
    #
    bf_num_inter = b
    bf_step = step_size
    s=sequence


    val1_set = set()

    rem_val1 = val1 % bf_step  # Convert into values within same interval
    if rem_val1 >= bf_step / 2:
        use_val1 = val1 + (bf_step - rem_val1)
    else:
        use_val1 = val1 - rem_val1



    val1_set.add(str("{0:.2f}".format(float(use_val1)))+str(sequence))  # Add the actual value


    # Add variations larger and smaller than the actual value
    #
    for i in range(bf_num_inter + 1):
        diff_val = (i + 1) * bf_step
        val1_set.add(str("{0:.2f}".format(use_val1 - diff_val))+str(sequence))


        diff_val = (i) * bf_step
        val1_set.add(str("{0:.2f}".format(use_val1 + diff_val))+str(sequence))

    return val1_set

# val_set
def BF_filter(val_set,num_hash,length):
    bloom_set=bitarray(length)
    bloom_set.setall(False)

    for val in val_set:
        hex_str1 = h1(val).hexdigest()
        int1 = int(hex_str1, 16)
        hex_str2 = h2(val).hexdigest()
        int2 = int(hex_str2, 16)

        for i in range(num_hash):
            gi = int1 + i * int2
            gi = int(gi % length)
            bloom_set[gi] = True

    return bloom_set



# hb_array a series of hb data record (60)
def convert_to_bf_input(hb_array):
    bf_input=set()
    # assert len(hb_array) <= 1, "only 1 data "
    length=len(hb_array)
    for i in range(length):
        # print hb_array[i],i
        bf_input = bf_input.union(add_neibour(hb_array[i],i))
        # bf_input.add(add_neibour(hb_array[i],i))  #add each one with neigbours into
    return bf_input

print type(hb_data)
bf_input = convert_to_bf_input(hb_data)
print type(hb_data1)
bf_input1 = convert_to_bf_input(hb_data1)

# print "bf_hb: ",bf_hb
# print "bf_hb1: ",bf_hb1
# print len(convert_to_bf_input(hb_data))

bf_hb=BF_filter(bf_input,num_hash,length)
bf_hb1=BF_filter(bf_input1,num_hash,length)

print 'afte bf is 1 ' ,bf_hb
print 'afte bf is 2 ' ,bf_hb1

# cal the difference/distance of two bitarray(after BF)
def cal_diff_two_bf(bf1,bf2):
    comm_bits = (bf1 & bf2).count()
    distance = 2 * float(comm_bits) / (int(bf1.count())+int(bf2.count()))
    assert distance >= 0.0 and distance <= 1.0, (distance)
    return distance
#
# bf_hb=BF_filter(bf_input,num_hash,length)
# bf_hb1=BF_filter(bf_input1,num_hash,length)

dis_bf_hb=cal_diff_two_bf(bf_hb,bf_hb1)
print 'difference is: ',dis_bf_hb

def add_neibour_for_two( val1, val2):
    """Calculate absolute difference similarity between two
       numerical values encoded into Bloom filters.
    """

    # Number of intervals and their sizes (step) to consider
    #
    bf_num_inter = b
    bf_step = step_size

    val1_set = set()
    val2_set = set()

    rem_val1 = val1 % bf_step  # Convert into values within same interval
    if rem_val1 >= bf_step / 2:
        use_val1 = val1 + (bf_step - rem_val1)
    else:
        use_val1 = val1 - rem_val1

    rem_val2 = val2 % bf_step
    if rem_val2 >= bf_step / 2:
        use_val2 = val2 + (bf_step - rem_val2)
    else:
        use_val2 = val2 - rem_val2

    val1_set.add(str(float(use_val1)))  # Add the actual value
    val2_set.add(str(float(use_val2)))  # Add the actual value

    # Add variations larger and smaller than the actual value
    #
    for i in range(bf_num_inter + 1):
        diff_val = (i + 1) * bf_step
        val1_set.add(str(use_val1 - diff_val))
        val2_set.add(str(use_val2 - diff_val))

        diff_val = (i) * bf_step
        val1_set.add(str(use_val1 + diff_val))
        val2_set.add(str(use_val2 + diff_val))

    return val1_set, val2_set

# val1_nei,val2_nei=add_neibour_for_two(66,67)
# print type(val2_nei)
# print val1_nei
# print val2_nei

val1_nei= add_neibour(66,'a')
# print len(val1_nei)





# print BF_filter(val1_nei,num_hash,length)

#-----compress------
# ## generate the Guassian matrix
# # mu, sigma = 0, 0.1
# # s = np.random.normal(mu, sigma, (5,60))
# # print s.shape
# # np.save('random_matrix',s)
#
# # load fixed random matrix
# random_m=np.load('random_matrix.npy')
# # print random_m
#
# hb_data=series[:60][1]
# compress_hb=np.matmul(random_m,hb_data)
# print compress_hb
# # print bitarray(compress_hb)