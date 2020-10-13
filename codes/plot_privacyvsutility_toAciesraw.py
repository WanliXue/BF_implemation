import matplotlib as mpl
mpl.use('TkAgg')  #for mac using matplotlib
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 20})


# # har
#
# coor_base=[0.99, 0.99, 0.95, 0.79]
# # coor_base=[0.83, 0.81, 0.80, 0.51]
# base_utility=[ 90.5,	  87.48,	88.39	,	80.73 ]
#
# our =         [90.8,	90.3,	88,	89.3,	87,	85,	84.5,	84.5]
# coor_our=[0.9, 0.80, 0.70,0.60, 0.51, 0.40, 0.30, 0.21]
# # coor_our=[0.81, 0.722, 0.6365,0.5365, 0.4499, 0.3617, 0.2776, 0.1772]
#
#
# plt.plot(coor_base,base_utility,'-r.',label='baseline')
# plt.plot(coor_our,our,'-b',label='FP=0.1')
# plt.legend(loc='lower left')
# plt.ylabel('Accuracy(%)')
# plt.xlabel('Coefficient(r)')
# plt.show()


# # mnist
#
#
# coor_base=[ 0.98, 0.96, 0.81, 0.57, 0.50]
# #coor_base=[ 0.535, 0.49, 0.43, 0.25]
# base_utility=[90.40 , 90.80,	88.30 , 80.10, 53]
#
# our =         [92.4	,90.9	,89.7, 86.2	,83	,80.9	,75.5	,68.4]
# coor_our=[0.78, 0.699, 0.611,0.519, 0.4373, 0.3492, 0.2567, 0.1804]
# coor_our=[0.90, 0.80, 0.70, 0.62, 0.51, 0.41, 0.31, 0.22]
# # plt.plot(x,result_absolu,x,result_bf)
#
# plt.plot(coor_base,base_utility,'-r.',label='baseline')
# plt.plot(coor_our,our,'-b',label='FP=0.1')
# plt.legend(loc='lower left')
# plt.ylabel('Accuracy(%)')
# plt.xlabel('Coefficient(r)')
# plt.show()


# csi

coor_base=[0.99, 0.99, 0.99, 0.94]
# coor_base=[0.90, 0.90, 0.89, 0.83]
base_utility=[82.00,	64  ,  36 ,	16]

our =      [92,	90,	90	,88	,84	,82,80	,56]
coor_our=[0.90, 0.80, 0.70, 0.60, 0.50, 0.41, 0.30, 0.21]
# plt.plot(x,result_absolu,x,result_bf)

plt.plot(coor_base,base_utility,'-r.',label='baseline')
plt.plot(coor_our,our,'-b',label='FP=0.1')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
plt.xlabel('Coefficient(r)')
plt.show()