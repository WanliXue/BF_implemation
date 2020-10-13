import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# data_train=np.load('/Users/wanli/Desktop/bf_deep/cifar-data/cifar_batch1_bfed.npy')
#
#
# data_train=np.reshape(data_train,(9000,30000))
# print(data_train.shape)
#
# data_test = np.load('/Users/wanli/Desktop/bf_deep/cifar-data/cifar_test_bfed.npy')
#
# data_test = np.reshape(data_test,(5000,30000))
# print(data_test.shape)
#
# label_test = np.load('/Users/wanli/Desktop/bf_deep/cifar-data/batch_test_label.npy')
# print(len(label_test))
#
# label_train = np.load('/Users/wanli/Desktop/bf_deep/cifar-data/batch_1_train_label.npy')
# print(len(label_train))
#
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(data_train,label_train.ravel())
# predict_result=knn.predict(data_test)
# # print predict_result
# print 'knn acuracy is ',accuracy_score(label_test.ravel(),predict_result)

#
data_train=np.load('/Users/wanli/Dropbox/ppml_code_with_dataset/CIFAR/cifar_batch1_compress300.npy')
print (data_train.shape)
data_train = np.reshape(data_train,[9000,900])
print (data_train.shape)
#
label_train = np.load('/Users/wanli/Desktop/bf_deep/cifar-data/batch_1_train_label.npy')
print(len(label_train))


np.random.seed(42)
rndperm = np.random.permutation(data_train.shape[0])

data = data_train[rndperm,:]
print(data.shape)

label = label_train[rndperm]

train = data[0:8500,:]
print(train.shape)
train_label = label[0:8500]


test = data[8500:,:]
print(test.shape)
test_label = label[8500:]

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train,train_label)
predict_result=knn.predict(test)
# print predict_result
print 'knn acuracy is ',accuracy_score(test_label,predict_result)