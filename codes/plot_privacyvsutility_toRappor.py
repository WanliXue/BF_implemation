import matplotlib as mpl
mpl.use('TkAgg')  #for mac using matplotlib
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 20})


ymin=40
ymax=100

# #har
# coor_base=[0.90, 0.80, 0.70, 0.61, 0.51, 0.41, 0.31, 0.21]
# base_utility=[90.7	,90.3,	87.3,	87.4,	83.3	,85,	85	,81]
# our =         [90.8,	90.3,	88,	89.3,	87,	85,	84.5,	84.5]
# coor_our=[0.9, 0.80, 0.70,0.60, 0.51, 0.40, 0.30, 0.21]
# # plt.plot(x,result_absolu,x,result_bf)
#
# plt.plot(coor_base,base_utility,'-r.',label='baseline')
# plt.plot(coor_our,our,'-b',label='FP=0.1')
# plt.legend(loc='lower left')
# plt.ylabel('Accuracy(%)')
# plt.xlabel('Coefficient(r)')
# plt.ylim(ymin, ymax)
# plt.show()

# mnist

# coor_base=[0.90, 0.80, 0.70, 0.62, 0.51, 0.41, 0.31, 0.22]
# base_utility=[91.4,	90.8,	89.9,	87.7,	85.4	,80.2,	75.9,	72.6]
# our =         [92.4	,90.9	,89.7,86.2	,83	,80.9	,75.5	,68.4]
# # coor_our=[0.78, 0.699, 0.611,0.519, 0.4373, 0.3492, 0.2567, 0.1804]
# coor_our=[0.90, 0.80, 0.70, 0.62, 0.51, 0.41, 0.31, 0.22]
#
# # plt.plot(x,result_absolu,x,result_bf)
#
# plt.plot(coor_base,base_utility,'-r.',label='baseline')
# plt.plot(coor_our,our,'-b',label='FP=0.1')
# plt.legend(loc='lower left')
# plt.ylabel('Accuracy(%)')
# plt.xlabel('Coefficient(r)')
# plt.ylim(ymin, ymax)
# plt.show()

# csi

coor_base=[0.90, 0.80, 0.70, 0.60, 0.50, 0.41, 0.30, 0.21]
base_utility=[94,	88	,86	,84	,76,	86,	58	,48]
our =      [92,	90,	90	,88	,84	,82,80	,56]
#coor_our=[0.542, 0.484, 0.417,0.37, 0.303, 0.246, 0.178, 0.119]
coor_our=[0.90, 0.80, 0.70, 0.60, 0.50, 0.41, 0.30, 0.21]
# plt.plot(x,result_absolu,x,result_bf)

plt.plot(coor_base,base_utility,'-r.',label='baseline')
plt.plot(coor_our,our,'-b',label='FP=0.1')
plt.legend(loc='lower left')
plt.ylabel('Accuracy(%)')
plt.xlabel('Coefficient(r)')
plt.ylim(ymin, ymax)
plt.show()