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


# data=scipy.io.loadmat('../data/har_eps8.mat')
# data=scipy.io.loadmat('../data/har_eps5.mat')
# data=scipy.io.loadmat('../data/har_eps2.mat')
# data=scipy.io.loadmat('../data/har_epsln2.mat')
data=scipy.io.loadmat('../data/har_epsln1_2.mat')


# data=scipy.io.loadmat('../data/mnist_eps8.mat')
# data=scipy.io.loadmat('../data/mnist_eps5.mat')
# data=scipy.io.loadmat('../data/mnist_eps2.mat')
# data=scipy.io.loadmat('../data/mnist_epsln2.mat')

# data=scipy.io.loadmat('../data/csi_eps8.mat')
# data=scipy.io.loadmat('../data/csi_eps5.mat')
# data=scipy.io.loadmat('../data/csi_eps2.mat')
# data=scipy.io.loadmat('../data/csi_epsln2.mat')

# print data['test_set'].shape #(1000,784)

raw=np.array(data['har_raw'])

noise=np.array(data['har_noise'])

# raw=np.array(data['raw'])
# # har_raw=har_raw.transpose()
# noise=np.array(data['noise'])
# # har_noise=har_noise.transpose()

# com_to_len = 50
# length = 10000
# b =5
# num_hash = 10
# dis = float(5)
# g=(2*b+2)*com_to_len
# false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )

com_to_len = 60
length = 10000
b =10
num_hash = 5
dis = float(5)
g=(2*b+2)*com_to_len
false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )

print 'lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive





false_positive= math.pow( 1-math.exp(-(num_hash*g)/length) , num_hash )

bf_ts=BF_TS(length,num_hash,b,dis/(2*b),dis)

pair_number=10
# generate 100 pairs in 7352elements
# pair_array = bf_ts.generate_random_pairs(7352,pair_number)  # for har
# pair_array = bf_ts.generate_random_pairs(6000,pair_number)  # for mnist
pair_array = bf_ts.generate_random_pairs(150,pair_number)  # for csi


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



# --------  bf_dp_distance  2 noise---------

# bf_dp distance calcu
#
dataA=np.random.permutation(dataA)

har_train1=raw[dataA]
har_train2=noise[dataA]


cor=[]

#
cor=corr2_coeff(har_train1,har_train2)
# bfdp_difference.append('%.3f' %    (bf_ts.cal_diff_two_bf   (csbfdp_set1,csbfdp_set2)  )  )

print cor.shape

import seaborn as sns
corr = cor

print(np.sort(corr.flatten())[:1])
print(np.sort(corr.flatten())[99:100])

if np.sort(corr.flatten())[:1] <=0:
    min=0
else:
    min=np.sort(corr.flatten())[:1]
print (min+np.sort(corr.flatten())[99:100])/2

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