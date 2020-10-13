import numpy as np
import scipy.io



batch =1
i=0
# filename = '/Users/wanli/Desktop/bf_deep/saved_only_neighbours/bfed_batch' + str(batch) + '_' + str(i) + '.npy'
# filename = '/Users/wanli/Desktop/bf_deep/saved_only_neighbours/bfed_batch'+str(batch)+'_'+ str(i) + '_dis05.npy'
# filename = '/Users/wanli/Desktop/bf_deep/saved_only_neighbours/bfed_batch'+str(batch)+'_'+ str(i) + '_dis005.npy'


# filename = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/svd_har_dis5.npy'
# filename = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/svd_har_dis05.npy'
# filename = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/svd_har_dis005.npy'

# filename = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/har_dis005.npy'



# filename = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/yaleb_dis005.npy'

# filename='/Users/wanli/Desktop/sanjay_mnist/data/mnist_test_inputdata_full.npy' #(7352,561)

filename = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/mnist_dis00025.npy'

chan = np.load(filename)

print(type(chan))
print(len(chan[0]))
print(len(chan))


chan = np.asarray(chan)
chan = chan.astype(float)
print(chan.shape)
print(chan[0][0])
print(type(chan[0][0]))





#save to matlab file
adict= {}
adict['cha1'] = chan
# str = '/Users/wanli/Desktop/bf_deep/saved_only_neighbours/bfed_batch' + str(batch) + '_' + str(i) + '.mat'
# str = '/Users/wanli/Desktop/bf_deep/saved_only_neighbours/bfed_batch' + str(batch) + '_' + str(i) + '_dis05.mat'
# str = '/Users/wanli/Desktop/bf_deep/saved_only_neighbours/bfed_batch' + str(batch) + '_' + str(i) + '_dis005.mat'

# str = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/svd_har_dis5.mat'
# str = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/svd_har_dis05.mat'
# str = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/svd_har_dis005.mat'

# str = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/har_dis005.mat'

# str = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/yaleb_dis005.mat'

# str = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/mnist_raw.mat'

str = '/Users/wanli/Dropbox/ppml_code_with_dataset/distribution_plot_about_paras/mnist_dis00025.mat'

scipy.io.savemat(str,adict)