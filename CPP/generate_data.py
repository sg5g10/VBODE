import lvssa
import numpy as np
import matplotlib.pyplot as plt
import time

def test_add(times):
    x_prev = np.array([100,100])
    ssa = []
    c = np.array(np.array([0.53, 0.0025, 0.3]))
    for i in range(1,len(times)):

        x_next = np.array(lvssa.LV(c, x_prev, times[i-1], times[i]))
        ssa.append(x_next)
        x_prev = x_next
    ss = np.array(ssa)
    x0 = np.array([100,100]).reshape((1,2))
    ss = np.concatenate((x0, ss),axis=0)
    plt.plot(times, ss)
    print(ss)
  

if __name__ == '__main__':

    start = time.time()
    
    times = np.arange(0,50,5)
    ssa = []
    test_add(times)
    end = time.time()
    print(end - start)
