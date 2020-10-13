import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  accuracy_score
from sklearn.ensemble import RandomForestClassifier

har_train = np.load('../data/HAR_train_inputdata.npy')
har_test = np.load('../data/HAR_test_inputdata.npy')
label_train=np.load('../data/HAR_trainlabel.npy') #(7352,)
label_test=np.load('../data/HAR_testlabel.npy') #(2947,)


# print(har_train.shape)
# print(har_test.shape)
# print(label_test.shape)
# print (label_train.shape)








model = RandomForestClassifier(n_estimators=20,
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
model.fit(har_train,  label_train.ravel())
y_pred = model.predict(har_test)

accuracy = accuracy_score(label_test.ravel(),y_pred)
# print accuracy_score(label_test.ravel(),predict_result) # 0.86
print ('random tree',accuracy)


