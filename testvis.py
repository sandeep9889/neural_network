import matplotlib.pyplot 
# %matplotlib inline
import numpy as np
a = np.zeros([3,2])
a[0,0] = 1
a[0,1] = 2
a[1,0] = 9
a[2,1] = 12
print(a)
matplotlib.pyplot.imshow(a,interpolation="nearest")
