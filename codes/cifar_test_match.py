import time
from sklearn.manifold import TSNE
import numpy as np
import math
import matplotlib as mpl
mpl.use('TkAgg')  #for mac using matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import pandas as pd
from matplotlib import cm

import seaborn as sns
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})
# RS = 123

mpl.rcParams.update({'font.size': 20})

# from BF_TS import BF_TS
#
# data_train=np.load('/Users/wanli/Desktop/bf_deep/cifar-data/cifar_batch1_bfed.npy')
# data_train=np.reshape(data_train,(9000,30000))
# print(data_train.shape)
#
#
# raw_data = np.load('/Users/wanli/Desktop/bf_deep/cifar-data/batch_1.npy')
# raw_data = np.reshape(raw_data,(808,3072))
# print(raw_data.shape)
# #
# pair_number=100
#
# length = 10000
# b=10
# num_hash = 5
# dis = float(5)
# bf_ts=BF_TS(length,num_hash,b,dis/(2*b),dis)
#
# pair_array = bf_ts.generate_random_pairs(800,pair_number)
#
# actual_distance=[]
# bf_difference=[]
# method=0  #use what to calcualte the actual distance 0:uclidean 1:pearson 2:spearman
#
#
# #------------ for  actual distance vs bf distance----------------
# for i in range(pair_number):
#
#     data1 = raw_data[pair_array[i][0],:]
#     data2 = raw_data[pair_array[i][1],:]
#     actual_distance.append('%.3f' % (bf_ts.cal_abs_diff(data1, data2, method)))
#
#     # bf_CS distance
#
#     bf_data1=data_train[pair_array[i][0],:]
#     bf_data2=data_train[pair_array[i][1],:]
#
#
#     bf_difference.append('%.3f' %    (bf_ts.cal_abs_diff   (bf_data1,bf_data2, method)  )  )
#
#
#
# absolu_dif = np.array(actual_distance).astype(np.float)
# bl_dif = np.array(bf_difference).astype(np.float)
#
# # normalize the result to 0-1 and decimal 3
# result_absolu=np.around(  (absolu_dif-np.min(absolu_dif))/np.ptp(absolu_dif), decimals=4  )
# result_bf=np.around(   (bl_dif-np.min(bl_dif))/np.ptp(bl_dif), decimals=4  )
#
# How_far_away= bf_ts.cal_abs_diff(result_absolu,result_bf,0)
#
# # -------plot-------
# x=np.arange(0.0,1.0,0.1)
#
# # plt.plot(x,result_absolu,x,result_bf)
#
# plt.plot(result_absolu,result_bf,'o')
# plt.plot(x,x)
# plt.ylabel('Actual distance similarity')
# plt.xlabel('Bloom filter based distance similarity')
# # plt.title('len = %s, b= %s, distance= %s, num_hash=%s, fp=%s, howfar=%s'%(length,b,dis,num_hash,false_positive,How_far_away))
# # plt.title('len = %s, b= %s, distance= %s, num_hash=%s, fp=%s, howfar=%s'%(length,b,dis,num_hash,false_positive,How_far_away))
# plt.show()
# #
#
# # ---------


# raw_data = np.load('/Users/wanli/Dropbox/ppml_code_with_dataset/CIFAR/cifar-batch1_raw.npy')
# raw_data=np.reshape(raw_data,[9000,3072])
# print(raw_data.shape)

raw_data = np.load('/Users/wanli/Dropbox/ppml_code_with_dataset/CIFAR/cifar_batch1_compress300.npy')
raw_data=np.reshape(raw_data,[9000,900])
print(raw_data.shape)

np.random.seed(42)
rndperm = np.random.permutation(raw_data.shape[0])


y=np.load('/Users/wanli/Desktop/bf_deep/cifar-data/batch_1_train_label.npy')

X=raw_data
feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]

df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))


N=1000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values



time_start = time.time()

# --------------TSNE1---
y_train = df_subset['label'].values
model = TSNE(n_components=2, random_state=0)
tsne_results = model.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

plt.figure(figsize=(8,8))
cmap = cm.get_cmap('jet', 10)
plt.scatter(x=tsne_results[:,0],y=tsne_results[:,1],c=y_train,s=35,cmap=cmap)
plt.title("t-sne on 300 pixelvalues cifar10")
plt.colorbar()
plt.show()
# -------------

# -----------------TSNE 2
# tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=5000)
# tsne_results = tsne.fit_transform(data_subset)
# print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
# df_subset['tsne-2d-one'] = tsne_results[:,0]
# df_subset['tsne-2d-two'] = tsne_results[:,1]
#
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="y",
#     palette=sns.color_palette("hls", 10),
#     data=df_subset,
#     legend="full",
#     alpha=0.3
# )
# plt.show()

# -------------