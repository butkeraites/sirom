import numpy as np
from smt.sampling_methods import LHS

xlimits = np.array([[0.0, 1.0]]*15)
#print(xlimits)
sampling = LHS(xlimits=xlimits)

num = 10000
x = sampling(num)

print(x)