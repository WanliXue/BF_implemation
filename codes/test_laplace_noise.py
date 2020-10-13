import numpy as np
import math
import matplotlib as mpl
mpl.use('TkAgg')  #for mac using matplotlib
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 20})
# loc  = 0.
# # scale =1.  # location = mean scale=decay
# # epsilon = math.log(2)
# epsilon = 5
# print 'epsilon: ',epsilon
# scale = float(1)/epsilon
# print 'scale: ', scale
#
#
#
# s = np.random.laplace(loc, scale, 10000)
#

# # print type(s)
# # print s.shape
# # print s[0]
# # print s[999]
# p=2.0
# t=p * 1/epsilon
# print t
#
# t_noise =s
#
# # this range not make change
# a=np.count_nonzero(t_noise >= p * 1 / epsilon)
# b=np.count_nonzero(t_noise <= - p * 1 / epsilon)
# print 'a+b:',a+b
# # a=t_noise.count[t_noise >= p * 1 / epsilon]
# # b=t_noise.count[t_noise <= - p * 1 / epsilon]
# print 'how many changed: ',10000 -(a+b)
#
#
# count, bins, ignored = plt.hist(s, 10, normed=True)
# x = np.arange(-1., 1., .01)
# pdf = np.exp(-abs(x-loc)/scale)/(2.*scale)
# plt.plot(x, pdf)
#
# plt.axvline(x=t,color='r')
# plt.axvline(x=-t,color='r')
# # g = (1/(scale * np.sqrt(2 * np.pi)) * np.exp(-(x - loc)**2 / (2 * scale**2)))
# # plt.plot(x,g)
#
# plt.show()




loc  = 0.  # location = mean
# scale =1.  # location = mean scale=decay
epsilon = 1
sensity =1.0
print 'epsilon: ',epsilon
scale = sensity/epsilon
print 'scale: ', scale
p=0.3  #noise possibility


s = np.random.laplace(loc, scale, 10000)

cdf = p / 2;
x_value = math.log(2 * cdf)  # will be negative
print x_value
print 'epsilon :',epsilon,', p-value, ', p

count, bins, ignored = plt.hist(s, 100, normed=True)
x = np.arange(-1., 1., .01)
pdf = np.exp(-abs(x-loc)/scale)/(2.*scale)
plt.xlim(-7.5,7.5)
plt.plot(x, pdf)
# plt.text(-0.5, -0.5, r'$\sum_{i=0}^\infty x_i$', fontsize=20)
plt.text(1.8, 0.3, '1',
         fontsize=20)
plt.text(-2.8, 0.3, '1',
         fontsize=20)
plt.text(-0.3, 0.3, '0',
         fontsize=20)
plt.axvline(x=x_value,color='r')
plt.axvline(x=-x_value,color='r')
# g = (1/(scale * np.sqrt(2 * np.pi)) * np.exp(-(x - loc)**2 / (2 * scale**2)))
# plt.plot(x,g)
plt.title('PDF plot with p valule = 0.3',fontsize=20)


plt.show()



