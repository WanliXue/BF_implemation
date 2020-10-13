import os
import pandas as pd
import numpy as np
import math
import matplotlib as mpl
import scipy.io
mpl.use('TkAgg')  #for mac using matplotlib
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 20})
from BF_TS import BF_TS

data=scipy.io.loadmat('../data/csi.mat')

data_train=data['train_set']
data_test = data['test_set']
label_train = data['train_label']
label_test = data['test_label']



print data_train.shape
print data_test.shape
compressed_data = data_train
compressed_test_data = data_test



com_to_len = 80
length = 10000
b =10
num_hash = 5
dis = float(5)
g=(2*b+2)*com_to_len


# com_to_len = 50
# length = 10000
# b =10
# num_hash = 20
# dis = float(5)
# g=(2*b+2)*com_to_len
# false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )
#
# print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive
#




bf_ts=BF_TS(length,num_hash,b,dis/(2*b),dis)

pair_number=10
# generate 100 pairs in 7352elements
pair_array = bf_ts.generate_random_pairs(150,pair_number)


dataA=[]
dataB=[]
for i in pair_array:
    dataA.append(i[0])
    dataB.append(i[1])



# ------------------- HEAT MAP----------------------
#
def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))


# #
# # ----------------bf_ no noise-----------------
# #
#
# cs_data1 = compressed_data[dataA]
#
#
#
# bf_train1 = bf_ts.convert_set_to_bf(cs_data1[:10])
#
#
#
# har_train1 =bf_ts.convert_bitarray_to_train_data(bf_train1,len(bf_train1),length)
# har_train2 =bf_ts.convert_bitarray_to_train_data(bf_train1,len(bf_train1),length)
#
#
#
# # har_train1=bf_ts.adding_lp_noise_to_bf_data(har_train1,epsilon,length , p)
# # har_train2=bf_ts.adding_lp_noise_to_bf_data(har_train2,epsilon,length , p)
#
# cor=[]
#
# #
# cor=corr2_coeff(har_train1,har_train2)
# # bfdp_difference.append('%.3f' %    (bf_ts.cal_diff_two_bf   (csbfdp_set1,csbfdp_set2)  )  )
#
# print cor.shape
#
# import seaborn as sns
# corr = cor
#
# print(np.sort(corr.flatten())[:1])
# print(np.sort(corr.flatten())[99:100])
#
#
# sns_plot=sns.heatmap(corr,
#             xticklabels=range(10),
#             yticklabels=range(10),
#             cmap=sns.cm.rocket_r,
#             # cmap="YlGnBu_r"
#             vmin=0,
#             vmax=1
#             )
# plt.title('False Positive = %s' % np.around(  false_positive, decimals=2  ))
# plt.show(block=True)




# # --------  bf_dp_distance  large fp---------
# # bf_dp distance calcu
#
#
#
# cs_data1 = compress_data[dataA]
#
#
# bf_ts2=BF_TS(length,num_hash,b,dis/(2*b),dis)
# bf_train1 = bf_ts2.convert_set_to_bf(cs_data1[:10])
#
#
#
# har_train1 =bf_ts2.convert_bitarray_to_train_data(bf_train1,len(bf_train1),length)
# har_train2 =bf_ts2.convert_bitarray_to_train_data(bf_train1,len(bf_train1),length)
#
#
# cor=[]
#
# #
# cor=corr2_coeff(har_train1,har_train2)
# # bfdp_difference.append('%.3f' %    (bf_ts.cal_diff_two_bf   (csbfdp_set1,csbfdp_set2)  )  )
#
# print cor.shape
#
# import seaborn as sns
# corr = cor
#
# print(np.sort(corr.flatten())[:1])
# print(np.sort(corr.flatten())[99:100])
#
#
# sns_plot=sns.heatmap(corr,
#             xticklabels=range(10),
#             yticklabels=range(10),
#             cmap=sns.cm.rocket_r
#             # cmap="YlGnBu_r"
#             )
# plt.title('False Positive = %s' % np.around(  false_positive, decimals=2  ))
# plt.show(block=True)



# --------  bf_dp_distance  2 noise---------

# bf_dp distance calcu
#

cs_data1 = compressed_data[dataA]


bf_ts2=BF_TS(length,num_hash,b,dis/(2*b),dis)
bf_train1 = bf_ts2.convert_set_to_bf(cs_data1[:10])
#

bf_train1 = bf_ts2.convert_set_to_bf(cs_data1[:10])
false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )
print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive

har_train1 =bf_ts2.convert_bitarray_to_train_data(bf_train1,len(bf_train1),length)
har_train2 =bf_ts2.convert_bitarray_to_train_data(bf_train1,len(bf_train1),length)

loc  = 0.  # location = mean
# scale =1.  # location = mean scale=decay
epsilon = 1
sensity =1.0
print 'epsilon: ',epsilon
scale = sensity/epsilon
print 'scale: ', scale
p=0.3  #noise possibility


print 'epsilon :',epsilon,', p-value, ', p


har_train1=bf_ts2.adding_lp_noise_to_bf_data(har_train1,epsilon,length , p)
# har_train2=bf_ts.adding_lp_noise_to_bf_data(har_train2,epsilon,length , p)

cor=[]

#
cor=corr2_coeff(har_train1,har_train2)
# bfdp_difference.append('%.3f' %    (bf_ts.cal_diff_two_bf   (csbfdp_set1,csbfdp_set2)  )  )

print cor.shape

import seaborn as sns
corr = cor

print(np.sort(corr.flatten())[:1])
print(np.sort(corr.flatten())[99:100])
print (np.sort(corr.flatten())[:1]+np.sort(corr.flatten())[99:100])/2

sns_plot=sns.heatmap(corr,
            xticklabels=range(10),
            yticklabels=range(10),
            cmap=sns.cm.rocket_r,
            # cmap="YlGnBu_r"
            vmin=0,
            vmax=1
            )
# plt.title('False Positive = %s , P-value= %s' % (np.around(  false_positive, decimals=2  ),p))
plt.show()
# sns_plot.savefig("output.png")

print pd.DataFrame(cor,columns=list('0123456789'))


a= list(map(max, cor))

print a
print max(a)
print min(a)

# print pd.DataFrame(cor,columns=list('0123456789'))

print np.mean(cor)