import matplotlib as mpl
mpl.use('TkAgg')  #for mac using matplotlib
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 20})



ymin=10
ymax=100


b=    [5 ,10 ,    15 ,  20 ,   25 ,  30]
knn=[90.4, 90.6 , 40.5, 43 , 66.1,  14.6]
svm=[89.3, 89.4 , 83.1, 84.4, 86.1,  64.8]

# plt.plot(x,result_absolu,x,result_bf)

plt.plot(b,knn,'-r.',label='KNN')
plt.plot(b,svm,'-b',label='SVM')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
plt.ylim(ymin, ymax)
plt.xlabel('$b$')

# plt.title('len = %s, b= %s, distance= %s, num_hash=%s, fp=%s, howfar=%s'%(length,b,dis,num_hash,false_positive,How_far_away))
# plt.title('len = %s, b= %s, distance= %s, num_hash=%s, fp=%s, howfar=%s'%(length,b,dis,num_hash,false_positive,How_far_away))
plt.show()
#

proj=[20, 60 ,   100, 150 ,200, 300 ]
knn=[92.3 ,91  , 80 , 31  ,14 , 8]
svm=[89.1, 90.2, 87.6 ,81.6 ,68, 12.4]

plt.plot(proj,knn,'-r.',label='KNN')
plt.plot(proj,svm,'-b',label='SVM')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
plt.ylim(ymin, ymax)
plt.xlabel('$M$')
# plt.xlabel('Projection Dimension')
plt.show()


dist=[5   , 10  ,  15   , 20   ,30   , 40   , 50 ]
knn=[90.5 , 85.3 , 78.7 , 56.1 ,40.7 , 23.  , 9.5]
svm=[90.1 , 87.2 , 86.1 , 86.4 ,77.7 , 74.7 , 73.4]

plt.plot(dist,knn,'-r.',label='KNN')
plt.plot(dist,svm,'-b',label='SVM')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
plt.ylim(ymin, ymax)
plt.xlabel('$d_{max}$')
# plt.xlabel('Distance')
plt.show()

num_hash=[5, 10  ,20  ,    30 , 40 ,  50  ,  60]
knn=[90.3  ,91  , 78.3 ,  28.6 ,12.1, 12.3 , 9.7]
svm=[89.7 , 89.7, 87.6 ,  84.5 ,72.6, 41.8 , 13.4]
plt.plot(num_hash,knn,'-r.',label='KNN')
plt.plot(num_hash,svm,'-b',label='SVM')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
plt.ylim(ymin, ymax)
plt.xlabel('$k$')
# plt.xlabel('Number of Hash')
plt.show()


len=[5000 ,10000 ,15000 ,20000 ,30000 ,40000]
knn=[72  , 89.5 , 92.3 , 92.4 , 93   , 91.8]
svm=[86.8, 89.9,  90.1 , 90.2,  89.7,  89.6]
plt.plot(len,knn,'-r.',label='KNN')
plt.plot(len,svm,'-b',label='SVM')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
plt.ylim(ymin, ymax)
plt.xlabel('$m$')
# plt.xlabel('Length')
plt.show()



p_value=[0.05, 0.1  , 0.15 ,0.2  , 0.25 ,0.3    ,0.35 ]
knn=    [92.8 , 91.1 , 91  , 89   , 86.3 , 77   , 66.3]
svm=   [ 92.1 , 91.7  ,89.7 ,87.9 , 85.3 , 82.4 , 78.2]
plt.plot(p_value,knn,'-r.',label='KNN')
plt.plot(p_value,svm,'-b',label='SVM')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
plt.ylim(ymin, ymax)
plt.xlabel('$p$-value')
# plt.xlabel('p_value')
plt.show()

