from scipy.spatial.distance import pdist
import scipy
import numpy as np
r1=[True, False, True, False ,False, False]

z=scipy.spatial.distance.squareform(r1)
print z.shape
print z



iu1 = np.triu_indices(4)
print iu1

z[iu1]= False

print z
equal_pairs=np.argwhere(z==True)
print equal_pairs