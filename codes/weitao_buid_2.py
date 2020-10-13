from BF_TS_2digit_neigdis import BF_TS
import scipy.io
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import math

curr_time_tuple = time.localtime()






from numpy import genfromtxt
my_data = genfromtxt('/Users/wanli/Desktop/weitao/data_new.csv', delimiter=',')

# print(my_data.shape)

# label = my_data[:,0]
data = my_data[:,:]

# X_train, X_test, y_train, y_test = train_test_split( data, label, test_size=0.4, random_state=0)
#
# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
# print(clf.score(X_test, y_test))
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(clf, data, label, cv=10)
# print(scores)



# print(label)
print(data)
#
for i in range(1,11):
#
    # ------- BF data--------
    length = 500 #(300)
    print(length)
    b = 5  #(5)
    num_hash = 2  #2
    dis = float(i*0.01)   #0.03
    g=(2*b+2)
    false_positive= math.pow( 1-math.exp(-(float)(num_hash*g)/length) , num_hash )
    print ('lenth:',length,'b:',b,'num_hash:',num_hash,'dis:',dis ,'false_positive: ', false_positive, 'stepdis',dis)

    bf_ts=BF_TS(length,num_hash,b,dis,-1000)


    print 'BF start'
    # start_time=time.time()
    bf_train = bf_ts.convert_set_to_bf(data)  # the result it a list and hard to convert to np array
    print 'BF filter done'


    bfdata_train =bf_ts.convert_bitarray_to_train_data(bf_train,len(bf_train),length)
    print(bfdata_train)

    savepath = '/Users/wanli/Desktop/weitao/after/data_bfed_'+str(i)+'.csv'
    print(savepath)

    np.savetxt(savepath, (bfdata_train), delimiter=',')