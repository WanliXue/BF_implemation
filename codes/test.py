# from bitarray import bitarray
# import hashlib
# import math
# import numpy as np
# import os
# import pandas
# import matplotlib.pyplot as plt
#
# # a = bitarray()            # create empty bitarray
# # a.append(True)
# # a.extend([False, True, True])
# # a=10
# # print a
# a=bitarray(10)
# a.setall(False)
# # print a
#
# nei_list=[]
#
# num_attr_infor=['num', 29.0, 77.0, 2.4000000000000004, 17, 2000]
# print num_attr_infor[3]
#
# h1 = hashlib.sha1
# h2 = hashlib.md5
#
# length = 2000
# b = 5 #10
#
# # --hasnumber depends on neibouour list
# # num_hash = math.ceil(float(length/len(set(nei_list))) * numpy.log(2))
#
# name='Andrew'
# database = '..'+os.sep+'data'+os.sep+'data_%s.txt' % name
#
# ##load time series data
# series = pandas.read_csv(database,header=-1,delimiter='\t')
# i=60
# #get top i as one record
# hb_data=series[:i][1]  #only get first i data
# hb_data1=series[i:i+i][1]
#
#
# def plot_data(val1,val2):
#     x=np.arange(1,61,1)
#     plt.plot(x,val1,x,val2)
#     plt.show
#
# plot_data(hb_data,hb_data1)
# plt.ylabel('Heartrate')
# plt.show()
#
#
# t=float(13.3457)
# print t,type(t)
# tt="{0:.2f}".format(t)
# print tt
# st=str(t)+'a'
# print st, type(st)
#
# # assert type(t) is not float, "BLA"
#
# # --generate neibours list
# # this_attr_vals = []
# # attr_type = ''
# # for rec in rec_dict.values():
# #     this_attr_val = rec[attr]
# #     if self.isfloat(this_attr_val) or self.isint(this_attr_val):
# #         this_attr_val = float(this_attr_val)
# #         this_attr_vals.append(this_attr_val)
#
# # d = 0.05*(max(this_attr_vals) - min(this_attr_vals))
#
# # calculate n - len(set(nei_list))
# # uniq_elem_list = set(this_attr_vals)
# # for uniq_elem in uniq_elem_list:
# #     nei_list.append(uniq_elem)
# #     rem_uniq_elem_val = uniq_elem % (d/b)  # Convert into values within same interval
# #     if rem_uniq_elem_val >= 0.5*(d/b):
# #         nei_list.append(uniq_elem + ((d/b) - rem_uniq_elem_val))
# #     else:
# #         nei_list.append(uniq_elem - rem_uniq_elem_val)
# #
#
# bf_num_inter = 5
# bf_step = 0.24
#
# def bloom_test(val_set):
#     l=100
#     k=10
#     bloom_set = bitarray(l)
#     bloom_set.setall(False)
#
#     print "valset are" ,val_set
#
#     for val in val_set:
#         hex_str1 = h1(val).hexdigest()
#         int1 = int(hex_str1, 16)
#         hex_str2 = h2(val).hexdigest()
#         int2 = int(hex_str2, 16)
#
#         for i in range(k):
#             gi = int1 + i * int2
#             gi = int(gi % l)
#             print gi
#             bloom_set[gi] = True
#
#     return bloom_set
#
# # print bloom_test(['11','22','33'])
# # print bloom_test(['ab','bc','cd'])
#
#
# def nbf_calc_abs_diff( val1, val2):
#     """Calculate absolute difference similarity between two
#        numerical values encoded into Bloom filters.
#     """
#
#     # Number of intervals and their sizes (step) to consider
#     #
#     bf_num_inter = bf_num_inter
#     bf_step = bf_step
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
#
#
#
# # hb_set=set()
# # hb_set.add(str(integer))
# #
#
#
# # import pandas as pd
# #
# # d = {'Product1': {1:10, 45:15, 173:9}, 'Product2': {1:11, 100:50, 173:10}}
# # df = pd.DataFrame(d).T
# # print df
#
# import os
# import numpy as np
# import string
# import pandas
# import re
#
#
# # name='Andrew'
# # database = '..'+os.sep+'data'+os.sep+'data_%s.txt' % name
# #
#
# ##load time series data
# # series = pandas.read_csv(database,header=-1,delimiter='\t')
# # print (series.head())
#
#
#
#
#
#
#
#
# #
# # rec_dict={}
# # rec_no=0
# # in_file=open(database)
# # for rec in in_file:
# #     rec=rec.split(',')
# #     clean_rec = map(string.strip, rec)
# #     # print type(clean_rec)
# #     # clean_rec=re.split(r'\t+',clean_rec)
# #     for i in clean_rec:
# #         i.split('\t')
# #         # print i
# #         # print type(i)
# #         rec_dict[rec_no]=i
# #     rec_no += 1
# #
# # print rec_no
# # # print rec_dict
# # print rec_dict[8624]
# # # print re.split(r'\t+',rec_dict[1])
#
#


import random

a=0.4
print (random.random())

for i in range(10):
    print i
    once = random.random()
    if (once<a):
        print once
        if(once > (float)(a/2)):
            print (float)(a/2)
            print True
    else:
        print('B')