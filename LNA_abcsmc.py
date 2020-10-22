from scipy.integrate import odeint
import numpy as np
from abcsmc.abcsmc import ABCSMC
import matplotlib.pyplot as plt
import pickle
import time as timer
from CPP import lvssa

#
# Define The SSA simulator
#
def ssa_lv(c):
    times = np.arange(0,50,5)
    x_prev = np.array([100,100])
    ss = list()
    
    for i in range(1,len(times)):
        x_next = np.array(lvssa.LV(c, x_prev, times[i-1], times[i]))
        ss.append(x_next)      
        x_prev = x_next
    ss = np.array(ss)
    x0 = np.array([100,100]).reshape((1,2))
    return np.concatenate((x0, ss),axis=0)

def simulator(_theta): 
    
    theta = _theta.copy()
    theta[1] = theta[1]/100
    states = []
    for i in range(10):
        states.append(ssa_lv(theta))               
    return states
#
# Define a simple distance metric 
#
def dist_metric(d,x):
    dist=np.zeros(len(x))

    for sims in range(len(x)):

        dist[sims]= np.linalg.norm(d - x[sims])                   
    return dist


if __name__ == '__main__':
    ###  Use data generated earlier ###
    Y = np.array([[100, 100],[279, 308],[24, 250],[45,  75],
    [373, 122],[22, 428],[24, 110],[106,  60],[368, 312],[ 16, 263]])

    ###  Define priors, simulator ###
    priors =  [('beta', [1,2]), ('halfnormal', [1]), ('beta', [2,1])]
    model_sim = simulator

    ###  Run ABC-SMC with 1000 particles, 10 repeat simulations, and alpha=0.1 ###
    t0 = timer.time()
    sampler = ABCSMC(3,1000,Y,10,priors,model_sim,dist_metric,[1e3,80],quantile=10)
    samples = sampler.sample()
    t1 = timer.time()
    print('Total time', t1-t0)
    param_filename = './results/lna_abc_ssa.p'
    pickle.dump(samples, open(param_filename, 'wb'))






