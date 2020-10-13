import math
# import matplotlib as mpl
# mpl.use('TkAgg')  #for mac using matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
#
# length = 10000
#
#
#
#
# b = 5
#
# dis = float(20)
#
#
#
# num_hash = [2, 4, 6, 8, 10, 12, 14]
#
# com_to_len = 80
# g=(2*b+2)*com_to_len
#
#
# # ----num_hash-----
# false_positive = np.zeros(len(num_hash))
# for i in range(len(num_hash)):
#
#     false_positive[i]= math.pow( 1-math.exp(-(num_hash[i]*g)/length) , num_hash[i] )
#
# print false_positive
#
#
# plt.plot(num_hash,false_positive)
# plt.ylabel('FP')
# plt.xlabel('num_hash')
# plt.title('len = %s, b= %s, distance= %s'%(length,b,dis))
# plt.show()
# #---------



# ---------------BF'ed data ML -----------
# length = 10000
# b = 5
# num_hash = 10
# dis = float(20)
#
#
#
# g=(2*b+2)*80
# false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )
# print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive
#
# print length/g * math.log(2)
#
#
# print 'e point:', float(num_hash*g)/length

print math.log(0.25/0.75)