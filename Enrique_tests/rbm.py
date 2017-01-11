
import numpy as np
import time

class RBM(object):
    def __init__(self, n_vis, n_hid, sigma=0.01):
        self.W = sigma * np.random.randn(n_vis,n_hid);
        self.vis_bias = np.zeros([n_vis,1])
        self.hid_bias = np.zeros([n_hid,1])
        self.n_vis = n_vis
        self.n_hid = n_hid

def sigmoid(vector):
    return 1 / (1 + np.exp(-vector))

def contrastive_divergence_rows_K(Xbatch, rbm, K, lr):
        
    batch_size = Xbatch.shape[0]

    Delta_W = np.zeros(rbm.W.shape)
    Delta_b = np.zeros([rbm.n_vis,1])
    Delta_c = np.zeros([rbm.n_hid,1])

    for i in range(batch_size):
        x =  Xbatch[i:i+1,:]
        xneg = Xbatch[i:i+1,:]

        for k in range(K):
            hneg = sigmoid( np.dot(xneg,rbm.W) + rbm.hid_bias.T) > np.random.rand(1,rbm.n_hid)
            xneg = sigmoid( np.dot(hneg,rbm.W.T) + rbm.vis_bias.T) > np.random.rand(1,rbm.n_vis)

        ehp = sigmoid( np.dot(x,rbm.W) + rbm.hid_bias.T)
        ehn = sigmoid( np.dot(xneg,rbm.W) + rbm.hid_bias.T)
        
        Delta_W += lr * ( x * ehp.T - xneg * ehn.T).T
        Delta_b += lr * (x - xneg).T
        Delta_c += lr * (ehp - ehn).T
    
    rbm.W += Delta_W / batch_size;
    rbm.vis_bias += Delta_b / batch_size;
    rbm.hid_bias += Delta_c / batch_size;
    
    return


X_train_rows = np.random.rand(5000,784);
X_batch_rows = X_train_rows[:200,:];
rbm = RBM(784,225)
start_time = time.time()
contrastive_divergence_rows_K(X_batch_rows, rbm, 1, 0.01)
print("--- %s seconds ---" % (time.time() - start_time))
