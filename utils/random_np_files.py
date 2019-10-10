import numpy as np

dir = "C:\\Users\\Jiawei\\Downloads\\tmp-updates\\"
n = 10

for i in np.arange(n):
    w = np.random.rand(2, 3)
    np.savetxt(dir + str(i), w)
