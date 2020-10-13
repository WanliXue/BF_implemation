import matplotlib as mpl
mpl.use('TkAgg')  #for mac using matplotlib
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 20})


ymin=0
ymax=100

b=    [5 ,10 ,    15 ,  20 ,   25 ,  30]
knn=[90, 88 , 52, 70 , 88,  26]
svm=[66, 72 , 38, 38, 56,  10]

# plt.plot(x,result_absolu,x,result_bf)

plt.plot(b,knn,'-r.',label='KNN')
plt.plot(b,svm,'-b',label='SVM')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
# plt.xlabel('b')
plt.ylim(ymin, ymax)
plt.xlabel('$b$')

# plt.title('len = %s, b= %s, distance= %s, num_hash=%s, fp=%s, howfar=%s'%(length,b,dis,num_hash,false_positive,How_far_away))
# plt.title('len = %s, b= %s, distance= %s, num_hash=%s, fp=%s, howfar=%s'%(length,b,dis,num_hash,false_positive,How_far_away))
plt.show()
#

proj=[30, 80 ,   100, 150 , 300 ]
knn=[84 ,92  , 78 , 82   , 12]
svm=[76, 74, 72 ,64 , 10]

plt.plot(proj,knn,'-r.',label='KNN')
plt.plot(proj,svm,'-b',label='SVM')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
# plt.xlabel('Projection Dimension')
plt.ylim(ymin, ymax)
plt.xlabel('$M$')
plt.show()


dist=[5   , 10  ,  15   , 20   ,35   , 45   , 55 ]
knn=[56 , 82 , 84 , 92 ,96 , 90  , 92]
svm=[26 , 38 , 52 , 64 ,70 , 68 , 70]

plt.plot(dist,knn,'-r.',label='KNN')
plt.plot(dist,svm,'-b',label='SVM')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
# plt.xlabel('Distance')
plt.ylim(ymin, ymax)
plt.xlabel('$d_{max}$')
plt.show()

num_hash=[ 5,    10  ,15  ,    20 , 28 ]
knn=[      90  , 84 ,  73 ,    54, 16 ]
svm=[      74,   56 ,  38 ,    12, 2 ]
plt.plot(num_hash,knn,'-r.',label='KNN')
plt.plot(num_hash,svm,'-b',label='SVM')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
# plt.xlabel('Number of Hash')
plt.ylim(ymin, ymax)
plt.xlabel('$k$')
plt.show()


len=[5000 ,10000 ,15000 ,20000 ,30000 ,40000]
knn=[84  , 90 , 90 , 90 , 92   , 88]
svm=[64, 74,  68 , 68,  66,  56]
plt.plot(len,knn,'-r.',label='KNN')
plt.plot(len,svm,'-b',label='SVM')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
# plt.xlabel('Length')
plt.ylim(ymin, ymax)
plt.xlabel('$m$')
plt.show()



p_value=[0.05, 0.1  , 0.15 ,0.2  , 0.25 ,0.3    ,0.35 ]
knn=    [92 ,   90 ,   90  , 89   , 84 , 82   ,   78]
svm=   [ 84 ,    80  ,  74 ,  60 ,   50 , 42 ,    34]
plt.plot(p_value,knn,'-r.',label='KNN')
plt.plot(p_value,svm,'-b',label='SVM')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
# plt.xlabel('p_value')
plt.ylim(ymin, ymax)
plt.xlabel('$p$-value')
plt.show()

