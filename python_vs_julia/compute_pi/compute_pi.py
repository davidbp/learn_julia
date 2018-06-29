import time
import random

def compute_pi(N):
    """
    Compute pi with a Monte Carlo simulation of N darts thrown in [-1,1]^2
    Returns estimate of pi
    """
    n_landed_in_circle = 0  # counts number of points that have radial coordinate < 1, i.e. in circle
    for i in range(N):
        x = random.random() * 2 - 1  # uniformly distributed number on x-axis
        y = random.random() * 2 - 1  # uniformly distributed number on y-axis
        r2 = x*x + y*y  # radius squared, in radial coordinates
        if r2 < 1.0:
            n_landed_in_circle += 1.

    return (n_landed_in_circle / N) * 4.0

N= 100000000
t0 = time.time()
print("Start Computing")
pi = compute_pi(N)
print("Approximate pi: {}".format(pi))
print( "Total time: {} seconds".format(abs(t0- time.time()))) 



