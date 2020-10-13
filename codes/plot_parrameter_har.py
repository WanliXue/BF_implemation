import matplotlib as mpl
mpl.use('TkAgg')  #for mac using matplotlib
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 20})





ymin=30
ymax=100

b  =[ 5    ,10  , 15   ,  20   ,  25    , 30]
svm =[88 ,  90.3 ,86.8 ,  87.5  , 87.5  , 81.4]
knn =[75.4, 81.2, 51.2 ,  80.7 ,  73.2 ,  38]


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

proj=[    30 ,50 ,100, 150   , 200   ,300]
knn =[78.4, 79.2 ,69  ,39.7  , 47	,	35.8]
svm	=[88.7 ,89.3, 89.1 ,88.1 , 87.2 , 80.9]


plt.plot(proj,knn,'-r.',label='KNN')
plt.plot(proj,svm,'-b',label='SVM')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
# plt.xlabel('Projection Dimension')
plt.xlabel('$M$')
plt.ylim(ymin, ymax)
plt.show()


dist=[   5 ,     10  ,  15    , 20  ,  30  ,   40 ,   50 ]
svm  =[88.2   ,85.5 , 82.3 ,  77.9 , 65.3  , 57.1 , 51]
knn  =[79.3  , 68   , 64   ,  57  ,  35  ,   34.9  ,31.6]

plt.plot(dist,knn,'-r.',label='KNN')
plt.plot(dist,svm,'-b',label='SVM')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
plt.ylim(ymin, ymax)
plt.xlabel('$d_{max}$')
plt.show()

num_hash=[  5  , 10   ,  20  ,   30  ,   40  ,   50  ,    60]
svm =[    88.1 , 90.2  , 90.5  , 86  ,   84.5 ,  79.2  ,  49.7]
knn  =   [83   , 80.8 ,  76   ,  60.8  , 37.6 ,  28.9  ,  16]

plt.plot(num_hash,knn,'-r.',label='KNN')
plt.plot(num_hash,svm,'-b',label='SVM')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
plt.ylim(10, ymax)
plt.xlabel('$k$')
plt.show()


len =[ 5000 ,10000 ,15000 ,20000 ,30000 ,40000]
svm  =[88.2, 89.1 , 87.6  ,88.5,  88.3,  88.6]
knn	=[ 76.2 ,78.5  ,76.9 , 80.2 , 78.2 , 77.5]
plt.plot(len,knn,'-r.',label='KNN')
plt.plot(len,svm,'-b',label='SVM')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
plt.ylim(ymin, ymax)
plt.xlabel('$m$')
plt.show()



p_value=[0.05, 0.1  , 0.15 ,0.2  , 0.25 ,0.3    ,0.35 ]
svm    =[ 90.7,  90.3 ,87.4 , 87.4   ,83.3 , 85   , 85.6   ]
knn    =[ 81.4 , 79.7 ,79.2 , 74.6  , 76.3 , 65.9,  57.6   ]
plt.plot(p_value,knn,'-r.',label='KNN')
plt.plot(p_value,svm,'-b',label='SVM')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
plt.ylim(ymin, ymax)
plt.xlabel('$p$-value')
plt.show()

