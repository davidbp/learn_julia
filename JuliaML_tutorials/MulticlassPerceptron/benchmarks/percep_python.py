
import sklearn
from sklearn import linear_model
import numpy as np
import time 

n_samples = 60000

X = np.random.rand(n_samples,784)
X = np.array(X, dtype="Float32")
y = np.random.randint(1,10,n_samples)

from sklearn.linear_model import Perceptron

p = Perceptron(n_iter=1)

t0 = time.time()
p.fit(X,y)
print("\ntime is:", time.time()-t0)
