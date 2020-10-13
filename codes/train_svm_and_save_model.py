# Save Model Using Pickle
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn import svm
from sklearn.metrics import accuracy_score
# import os
import time
# import pandas
import numpy as np
import math
from sklearn import metrics




# load data
data_train=np.load('../data/HAR_traindata.npy') #(7352,561)
data_test=np.load('../data/HAR_testdata.npy')  #(2947 , 561)
label_train=np.load('../data/HAR_trainlabel.npy') #(7352,)
label_test=np.load('../data/HAR_testlabel.npy') #(2947,)



print ("Start SVM")

d_train=data_train.tolist()
d_test=data_test.tolist()
c_train = label_train
c_test = label_test


clf = svm.SVC(kernel = 'linear')

clf.fit(d_train,c_train)


filename = 'face_recognition_model.sav'
pickle.dump(clf, open(filename, 'wb'))


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
y_pred = loaded_model.predict(d_test)
print("Accuracy:",metrics.accuracy_score(c_test, y_pred))


