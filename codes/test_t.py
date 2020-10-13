import math
import numpy as np

# from collections import Counter
#
# cnt = Counter()

# cnt.update('0.553')

# print(cnt)


thislist=[]
thislist.append('0.5133')
print(thislist[0])
print(type(thislist[0]))

a=float(thislist[0])
print(type(a))

print (a+0.22)

import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,
                help="name of the user")
args = vars(ap.parse_args())

# display a friendly message to the user
print("Hi there {}, it's nice to meet you!".format(args["name"]))


# a=np.zeros([4,2])
# b=np.ones([4,2])
# c=np.random.random([4,2])
#
# new= np.stack([a, b, c], axis=2)
# print(new)

# print (np.arange(0,10))

# w=list((3,3))+[int(3), 64]
# print(w)
#
# stri = [1] + list((1,1))+ [1]
# print(stri)

# cifar = np.load('../data/cifar_batch1_ch1_bfed.npy')

# cifar = np.load('../data/cifar_batch1_bfed.npy')
#
# print(cifar.shape)

# print(cifar[1][2][0])
#
# # new=np.reshape(cifar,(808,10000,3))
# # print(new.shape)
#
# # np.stack([arr1, arr2, arr3], axis=2)
#
# new=cifar.transpose(1, 2 ,0)
# print(new.shape)
#
# print(new[2][4][1])

# c1=np.array([[1 ,1],[1,1]])
# # print(c1.shape)
# c2=np.array([[2,2],[2,2]])
# c3=np.array([[3,3],[3,3]])
# c_t=np.array([c1,c2,c3])
# print (c_t.shape)
# print (c_t[2][1][1])
#
# # c_v=np.stack([c1, c2, c3], axis=2)
# # print (c_v.shape)
# # print(c_v)
# #
# c_tra=c_t.transpose(1,2,0)
# print(c_tra.shape)
# print(c_tra)
#
# print(c_tra[1][1][2])

# c_res=c_t.reshape(2,2,3)
# print(c_res)

# np.stack([arr1, arr2, arr3], axis=2)


# com_to_len = 60
#
# length = 10000
# b = 10
# num_hash = 15
# dis = float(5)
#
#
#
# g=(2*b+2)*com_to_len
# false_positive= math.pow( 1-math.exp(-(float)(num_hash*g)/length) , num_hash )
#
# print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive

# false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )
# print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive
#



# com_to_len = 50
# length = 10000
# b =10
# num_hash =30
# dis = float(5)
# g=(2*b+2)*com_to_len
# false_positive= math.pow( 1-math.exp(-(float)(num_hash*g)/length) , num_hash )
# print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive

# import hashlib
# import numpy as np
#
# h2=hashlib.sha1
# h1=hashlib.md5
#
# val = np.int8(12)
# # val=123
# print type(val)
#
# bf_len=100;
#
# hex_str1 = h1(val).hexdigest()
# print hex_str1
# int1 = int(hex_str1, 16)
# hex_str2 = h2(val).hexdigest()
# int2 = int(hex_str2, 16)
#
# for i in range(10):
#     gi = int1 + i * int2
#     gi = int(gi % bf_len)
#     print gi
