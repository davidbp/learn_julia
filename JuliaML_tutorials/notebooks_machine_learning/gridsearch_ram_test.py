import numpy as np
import sklearn
from sklearn import linear_model
from sklearn import model_selection

n_samples = 1000000

X = np.random.rand(n_samples,100)
y = np.random.randint(1,10,n_samples)

print("\n\tDoing CV Sequential way, monitor you ram\n\n")
clf = sklearn.linear_model.Perceptron()
grid_params = {"alpha":[0.0001]}
grid_clf = model_selection.GridSearchCV(clf, grid_params, cv=10, n_jobs=1, verbose=2)
grid_clf.fit(X,y)


print("\n\tDoing CV parallel way, monitor you ram\n\n")
clf = sklearn.linear_model.Perceptron()
grid_params = {"alpha":[0.0001]}
grid_clf = model_selection.GridSearchCV(clf, grid_params, cv=10, n_jobs=-1, verbose=2)
grid_clf.fit(X,y)


