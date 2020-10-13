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


mpl.rcParams.update({'font.size': 20})




# raw minst fashion data
raw_data = np.load('/Users/wanli/Dropbox/ppml_code_with_dataset/CIFAR_mnist/Fashion_train_random60.npy')
raw_data=np.reshape(raw_data,[6000,60])
y=np.load('/Users/wanli/Dropbox/ppml_code_with_dataset/CIFAR_mnist/Fashion_trainlabel_random60.npy')
print(y.shape)


# # bfed fanshion minst da
# ta after 60 random compress and bf (b=5, hash =10, distance = 5)
# raw_data = np.load('../data/fashion_bfed_train_random60.npy')
# y=np.load('/Users/wanli/Dropbox/ppml_code_with_dataset/CIFAR_mnist/label_train.npy')
# print(y.shape)


print(raw_data.shape)

np.random.seed(42)
rndperm = np.random.permutation(raw_data.shape[0])




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
plt.title("t-sne on bfed minst fashion")
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