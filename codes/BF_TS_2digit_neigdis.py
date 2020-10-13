import random
import hashlib
from scipy import spatial
import numpy as np
from scipy import stats
from bitarray import bitarray
import math
import scipy
from collections import Counter

class BF_TS:
    def __init__(self, bf_len, bf_num_hash_func, bf_num_inter, bf_step,
               max_abs_diff):

        self.bf_len = bf_len
        self.bf_num_hash_func = bf_num_hash_func
        self.bf_num_inter = bf_num_inter
        self.bf_step =round(bf_step,3)   # only keep 3 decimal
        # self.bf_step = 0.5

#         self.max_abs_diff = max_abs_diff not used?

        # self.min_val = min_val
        # self.max_val = max_val
        self.h1=hashlib.sha1
        self.h2=hashlib.md5

        # assert max_val > min_val

    # val1,2 are normal time series data (pdarray)
    def cal_abs_diff(self,val1, val2, method):
        if method == 0:  # euclidean distance
            d = spatial.distance.euclidean(val1, val2)
            return d
        if method == 3:  # pairwise distance bewteen ob in n-dimension
            # X=np.c_[val1, val2]
            X = np.array([val1, val2])
            y = spatial.distance.pdist(X.transpose(), 'seuclidean')
            d = spatial.distance.squareform(y)
            # d=spatial.distance.seuclidean(val1,val2)
            return d
        if method == 1:  # pearson corelation (-1,+1)
            [r, _] = stats.pearsonr(val1, val2)
            return r
        if method == 2:  # spearman  (-1,+1)
            [r, _] = stats.spearmanr(val1, val2)
            return r
        if method ==4:
            d=scipy.spatial.distance.jaccard(val1,val2)
            return d

    # calculate the bf difference with val1 val2 (bdarray, e.g., heartbeate)
    def cal_bf_diff(self, val1, val2):
        bf_input1 = self.convert_to_bf_input(val1)
        bf_input2 = self.convert_to_bf_input(val2)
        bf1=self.__set_to_bloom_filter__(bf_input1)
        bf2=self.__set_to_bloom_filter__(bf_input2)
        return self.cal_diff_two_bf(bf1,bf2)

##     used to calculate the extreme parameter
#     # calcualte number of hash function and maximum distance
#     def cal_BF_parrameter(hb_array, b, length):

#         dis = 0.05 * (max(hb_array) - min(hb_array))  # used to be 0.05

#         nei_list = []
#         uniq_elem_list = set(hb_array)
#         for uniq_elem in uniq_elem_list:
#             nei_list.append(uniq_elem)
#             rem_uniq_elem_val = uniq_elem % (dis / b)  # Convert into values within same interval
#             if rem_uniq_elem_val >= 0.5 * (dis / b):
#                 nei_list.append(uniq_elem + ((dis / b) - rem_uniq_elem_val))
#             else:
#                 nei_list.append(uniq_elem - rem_uniq_elem_val)
#         num_hash = int(math.ceil(float(length / len(set(nei_list))) * np.log(2)))

#         return num_hash, dis

    # calculate the bit difference of two bloom filter vector
    def cal_diff_two_bf(self,bf1, bf2):
        comm_bits = (bf1 & bf2).count()
        distance = 2 * float(comm_bits) / (int(bf1.count()) + int(bf2.count()))
        assert distance >= 0.0 and distance <= 1.0, (distance)
        return distance

    # covert bitaray{10000110} to np array [0, 1 ,...]
    def convert_bitarray_to_train_data(self,bf_data,bf_len,len):

        result_np = np.zeros(shape=(bf_len,len))

        # convert all bitarry to 0,1 features for train
        for i in range(bf_len):
            processing=bf_data[i]
            one_record=np.zeros(len)
            for t in range(len):
                one_record[t]=int(processing[t] ==True)

            result_np[i]=one_record
        # print one_record
        return result_np


    # covert bitaray{10000110} to np array [0, 1 ,...]
    # only for one bitarray used for distance testing
    def convert_one_bitarray_to_train_data(self,bf_data,bf_len,len):

        # result_np = np.zeros(shape=(bf_len,len))

        # convert all bitarry to 0,1 features for train

        processing=bf_data
        one_record=np.zeros(len)
        for t in range(len):
             one_record[t]=int(processing[t] ==True)

        return one_record





    # def nbf_calc_abs_diff(self, val1, val2):
    #     """Calculate absolute difference similarity between two
    #        numerical values encoded into Bloom filters.
    #     """
    #
    #     # Number of intervals and their sizes (step) to consider
    #     #
    #     bf_num_inter = self.bf_num_inter
    #     bf_step = self.bf_step
    #
    #     val1_set = set()
    #     val2_set = set()
    #
    #     rem_val1 = val1 % bf_step  # Convert into values within same interval
    #     if rem_val1 >= bf_step / 2:
    #         use_val1 = val1 + (bf_step - rem_val1)
    #     else:
    #         use_val1 = val1 - rem_val1
    #
    #     rem_val2 = val2 % bf_step
    #     if rem_val2 >= bf_step / 2:
    #         use_val2 = val2 + (bf_step - rem_val2)
    #     else:
    #         use_val2 = val2 - rem_val2
    #
    #     val1_set.add(str(float(use_val1)))  # Add the actual value
    #     val2_set.add(str(float(use_val2)))  # Add the actual value
    #
    #     # Add variations larger and smaller than the actual value
    #     #
    #     for i in range(bf_num_inter + 1):
    #         diff_val = (i + 1) * bf_step
    #         val1_set.add(str(use_val1 - diff_val))
    #         val2_set.add(str(use_val2 - diff_val))
    #
    #         diff_val = (i) * bf_step
    #         val1_set.add(str(use_val1 + diff_val))
    #         val2_set.add(str(use_val2 + diff_val))
    #
    #     return val1_set, val2_set


    # filtered with BF input is set of string
    def  __set_to_bloom_filter__(self,val_set):
        bloom_set = bitarray(self.bf_len)
        bloom_set.setall(False)

        for val in val_set:
#             print(type(val))
            val = str(val) #cast to string
            encode_val =val.encode('utf-8')
#hashlib.sha256(str(random.getrandbits(256)).encode('utf-8')).hexdigest()
            hex_str1 = self.h1(encode_val).hexdigest()
            int1 = int(hex_str1, 16)
            hex_str2 = self.h2(encode_val).hexdigest()
            int2 = int(hex_str2, 16)

            for i in range(self.bf_num_hash_func):
                gi = int1 + i * int2
                gi = int(gi % self.bf_len)
                bloom_set[gi] = True

        return bloom_set

    def convert_to_cs_bf_input_noneigibour(self,data): #data is dimension 50
        bf_input = []  #using list
        length = len(data)
        for i in range (length):
            bf_input.append(self.add_tolist(data[i]))
        return bf_input

#     normalise the data and add to the list for encoding to the bf
    def add_tolist(self, data):
        return round(data ,3) #single data
         
        
    # convert one data to bf_input
    def convert_to_cs_bf_input(self,data):
        bf_input = []
        length = len(data)
        # assert length <=5, 'too many'
        # st=['a','b','c','d','e']
        for i in range (length):
            # bf_input = bf_input.union(self.add_neibour(data[i],st[i]))
            # bf_input = bf_input.append(self.add_neibour(data[i], i))
            bf_input += self.add_neibour(data[i], i)
        return bf_input

    # hb_array a series of hb data record (60)
    def convert_to_bf_input(self, data):
        bf_input = set()
        # assert len(hb_array) <= 1, "only 1 data "
        length = len(data)
        for i in range(length):
            # print hb_array[i],i
            bf_input = bf_input.union(self.add_neibour(data[i], i))
            # bf_input.add(add_neibour(hb_array[i],i))  #add each one with neigbours into
        return bf_input

    # hb_array a series of hb data record (60)
    def convert_to_bf_input_withorder(self, data):
        bf_input = []
        # assert len(hb_array) <= 1, "only 1 data "
        length = len(data)
        for i in range(length):
            # print hb_array[i],i
            # this is for list []
            bf_input = bf_input.extend(self.add_neibour_with_order(data[i],i))

            # this is for set()
            # bf_input = bf_input.union(self.add_neibour(data[i], i))
            # bf_input.add(add_neibour(hb_array[i],i))  #add each one with neigbours into
        return bf_input

    # for a signel input 77
    # add neibour to raw value with time stamp {xx0,xx1,xx2,...,xxi}
    def add_neibour(self,val1, sequence):
        """Calculate absolute difference similarity between two
           numerical values encoded into Bloom filters.
        """

        # Number of intervals and their sizes (step) to consider
        #
        bf_num_inter = self.bf_num_inter
        bf_step = self.bf_step
        s = sequence

        val1_set = []

        rem_val1 = val1 % bf_step  # Convert into values within same interval
        rem_val1=round(rem_val1 ,3) #3 decimal  used to be 2
        if rem_val1 >= bf_step / 2:
            use_val1 = val1 + (bf_step - rem_val1)
        else:
            use_val1 = val1 - rem_val1

        # remove negative zero -0.00
        if use_val1 == 0:
            use_val1 = 0.000

        val1_set.append(str("{0:.3f}".format(float(use_val1))) + str(sequence))  # Add the actual value

        # Add variations larger and smaller than the actual value
        #
        for i in range(bf_num_inter):
            diff_val = (i + 1) * bf_step
            left_value=use_val1 - diff_val
            if float("{0:.3f}".format(left_value)) == 0:
                left_value = 0.000
            val1_set.append(str("{0:.3f}".format(left_value)) + str(sequence))

            diff_val = (i) * bf_step
            right_value = use_val1 + diff_val
            if float("{0:.3f}".format(right_value)) == 0:
                right_value = 0.000
            val1_set.append(str("{0:.3f}".format(right_value)) + str(sequence))

        return val1_set

    # add neibour to raw value with time stamp {xx0,xx1,xx2,...,xxi}
#     def add_neibour_with_order(self,val1, sequence):
#         """Calculate absolute difference similarity between two
#            numerical values encoded into Bloom filters.
#         """

#         # Number of intervals and their sizes (step) to consider
#         #
#         bf_num_inter = self.bf_num_inter
#         bf_step = self.bf_step
#         s = sequence

#         val1_set = []

#         rem_val1 = val1 % bf_step  # Convert into values within same interval
#         rem_val1=round(rem_val1 ,2) #2 decimal
#         if rem_val1 >= bf_step / 2:
#             use_val1 = val1 + (bf_step - rem_val1)
#         else:
#             use_val1 = val1 - rem_val1

#         # remove negative zero -0.00
#         if use_val1 == 0:
#             use_val1 = 0.000

#         val1_set.extend(str("{0:.2f}".format(float(use_val1))) + str(sequence))  # Add the actual value

#         # Add variations larger and smaller than the actual value
#         #
#         for i in range(bf_num_inter + 1):
#             diff_val = (i + 1) * bf_step
#             left_value=use_val1 - diff_val
#             if float("{0:.2f}".format(left_value)) == 0:
#                 left_value = 0.000
#             val1_set.extend(str("{0:.2f}".format(left_value)) + str(sequence))

#             diff_val = (i) * bf_step
#             right_value = use_val1 + diff_val
#             if float("{0:.2f}".format(right_value)) == 0:
#                 right_value = 0.0000
#             val1_set.extend(str("{0:.2f}".format(right_value)) + str(sequence))

#         return val1_set

    def generate_random_pairs(self,iter_range,num_pair):
        rand_val_pairs = []
        random.seed(42)  #fixed random
        for i in range(num_pair):
            v1 = random.randint(0, iter_range-1)

            v2 = random.randint(0, iter_range-1)
            rand_val_pairs.append([v1, v2])
        return rand_val_pairs

    
    def covert_set_to_bf_noneighbour(self,data):
        bf_data=[]
        # print data.ndim
        if data.ndim == 1:
            for i in range(len(data)):
                bf_input=self.convert_to_cs_bf_input_noneighbour(data[i])
                bf_tem_data=self.__set_to_bloom_filter__(bf_input)
                bf_data.append(bf_data,bf_tem_data)
        else:
            a,b = data.shape #a=7352 b = 50
            # print a,b
            for i in range(a):
                # print data[i].shape #(50,)
                bf_input = self.convert_to_cs_bf_input_noneigibour(data[i])
                bf_tem_data = self.__set_to_bloom_filter__(bf_input)
                bf_data.append(bf_tem_data)

        # result = np.array(bf_data, dtype=int)
        # return result
        return bf_data
    

    def only_store_neibour(self,data):
        bf_data = []
        # print data.ndim
        if data.ndim == 1:
            for i in range(len(data)):
                bf_input = self.only_get_neibou_list(data[i])
                bf_data.append(bf_input)
        else:
            a, b = data.shape  # a=7352 b = 50
            # print a,b
            for i in range(a):
                # print data[i].shape #(50,)
                bf_input = self.only_get_neibou_list(data[i])
                bf_data.append(bf_input)
        return bf_data


    def only_get_neibou_list(self, data):

        # bf_input = set()
        bf_input = []
        length = len(data)
        for i in range (length):
            one_data_list = self.add_neibour_withouttime(data[i])
            bf_input += one_data_list
            # bf_input = bf_input.union(self.add_neibour_withouttime(data[i]))
            # bf_input = bf_input.union(self.add_neibour(data[i], i))
        return bf_input

    def add_neibour_withouttime(self, val1):

        """Calculate absolute difference similarity between two
           numerical values encoded into Bloom filters.
        """

        # Number of intervals and their sizes (step) to consider
        #
        bf_num_inter = self.bf_num_inter
        bf_step = self.bf_step

        val1_set = []
        # val1_set = set()

        rem_val1 = val1 % bf_step  # Convert into values within same interval
        rem_val1 = round(rem_val1, 3)  # 3 decimal  used to be 2
        if rem_val1 >= bf_step / 2:
            use_val1 = val1 + (bf_step - rem_val1)
        else:
            use_val1 = val1 - rem_val1

        # remove negative zero -0.00
        if use_val1 == 0:
            use_val1 = 0.000
        raw_data = str("{0:.3f}".format(float(use_val1)))
        val1_set.append(raw_data)



        # Add variations larger and smaller than the actual value
        #
        for i in range(bf_num_inter):
            diff_val = (i + 1) * bf_step
            left_value = use_val1 - diff_val
            if float("{0:.3f}".format(left_value)) == 0:
                left_value = 0.000
            val1_set.append(str("{0:.3f}".format(left_value)) )

            diff_val = (i) * bf_step
            right_value = use_val1 + diff_val
            if float("{0:.3f}".format(right_value)) == 0:
                right_value = 0.000
            val1_set.append(str("{0:.3f}".format(right_value)) )

        return val1_set



    # convert a series data to bf_data
    def convert_set_to_bf(self,data):
        bf_data=[]
        # print data.ndim
        if data.ndim == 1:
            for i in range(len(data)):
                bf_input=self.convert_to_cs_bf_input(data[i])
                bf_tem_data=self.__set_to_bloom_filter__(bf_input)
                bf_data.append(bf_data,bf_tem_data)
        else:
            a,b = data.shape #a=7352 b = 50
            # print a,b
            if(self.bf_num_inter != 0):
                for i in range(a):
                    # print data[i].shape #(50,)
                    bf_input = self.convert_to_cs_bf_input(data[i])
                    bf_tem_data = self.__set_to_bloom_filter__(bf_input)
                    bf_data.append(bf_tem_data)
            else:
                for i in range(a):
                    bf_tem_data = self.__set_to_bloom_filter__(data[i])
                    bf_data.append(bf_tem_data)
        # result = np.array(bf_data, dtype=int)
        # return result
        return bf_data

    def convert_set_to_bf_intermediate(self,data):
        # convert input to data list with neigbours
        inter_data = []
        a,b = data.shape  #a=sample_number b = sample_dimension
        for i in range(a):
            print (i, ' in ',a)
            bf_input = self.convert_to_cs_bf_input_withorder(data[i])
            inter_data.append(bf_input)
        return inter_data

    def generate_lp_noise(self,epsilon,length):
        loc=0
        scale= float(1) / epsilon  # sensity/epsilon
        noise = np.random.laplace(loc, scale, length)
        return noise




    def adding_lp_noise_to_bf_data(self,data,epsilon, length , p):  # p can be regarded as threashold, larger then that we count
        # print 'data: ', data.shape

        # p= 0.9
        cdf = p / 2;
        x_value =  math.log(2 * cdf)  # will be negative
        print (x_value)

        a,b = data.shape
        noise_data = np.zeros(shape=(a, b))
        # print a
        for i in range(a):
            # print data[i].shape
            cur_data= data[i]


            # raw_noise = self.generate_lp_noise(epsilon,length)
            # t_noise=raw_noise

            t_noise = np.random.laplace(0, 1, length)

            # this range not make change
            # noise_a = np.count_nonzero(t_noise >= p * 1 / epsilon) # true
            # noise_b = np.count_nonzero(t_noise <= - p * 1 / epsilon)
            # print 'a+b:', noise_a + noise_b
            # print 'how many changed: ', length - (noise_a + noise_b)

            # old  and wrong
            # t_noise[t_noise >= p * 1/epsilon] = 1
            # t_noise[t_noise <=- p * 1 / epsilon] = 1
            # t_noise[(- p * 1 / epsilon <t_noise) & (t_noise < p * 1/epsilon)] = 0

            # # this order is import, zero frist than 1
            t_noise[(x_value < t_noise) & (t_noise < -x_value)] = 0
            t_noise[x_value >= t_noise] = 1
            t_noise[t_noise >= -x_value] = 1

#             print ('how many 1 in noise:', np.count_nonzero(t_noise))
#             print ('how many 1 raw data:', np.count_nonzero(cur_data))

            for temp in range(length):

                noise_data[i][temp] = int(cur_data[temp]) ^ int(t_noise[temp])

            # print noise_data[i]
#             print ('how many 1 after noise:', np.count_nonzero(noise_data[i]))
        return noise_data
