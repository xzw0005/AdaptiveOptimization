'''
Created on Oct 15, 2016

@author: Xing Wang
'''

'''
Created on Aug 29, 2016
@author: XING WANG    
'''

import numpy as np
import math
import time 

def evalCountOnes(x):
    return sum(np.array(x))
#     for xi in x:
#         val = 0
#         if (xi == 1):
#             val += 1
#         elif (xi != 0):
#             raise ValueError('Invalid Bit-Vector Input!')
#    return val   

def flipOneBit(x):
    xnew = np.copy(x)
    num_bits = len(x)
    i = np.random.randint(num_bits)
    xnew[i] = 1 - xnew[i]
    return xnew
    

class SA(object):
    '''
    Soving a CountOnes Problem, which has global minimum sum(x_i) when all x_i = 1
    '''
    
    def __init__(self, current_sol = None, init_temperature=1e2, m=1e3, n=1e1, alpha=.5, num_bits = 1e3, seed = None, maximize = True):
        '''
        Constructor
        '''
        num_bits = int(num_bits)
        self.maximize = maximize
        self.init_temperature = init_temperature
        self.m = m
        self.n = n
        self.alpha = alpha
        if (current_sol is not None):
            self.num_bits = len(current_sol)
            self.current_sol = current_sol
        else:
            self.num_bits = num_bits
            self.current_sol = self.getInitSolution(num_bits, seed)
        
    def getInitSolution(self, num_bits, seed):
        if (seed is not None):
            np.random.seed(seed)
        return np.random.randint(2, size=num_bits)
        
    def moveOperator(self, current_sol):
        return flipOneBit(current_sol)
    
    def evaluate(self, x):
        return evalCountOnes(x)
    
    def simulatedAnnealing(self):
        current_sol = self.current_sol
        temperature = self.init_temperature 
        alpha=self.alpha
        m=self.m
        n=self.n
        
        current_obj = self.evaluate(current_sol)
        iter = 0
        fit_hist = []
        while (iter < m) and (current_obj < len(self.current_sol)):
            iter += 1
            for i in np.arange(n - 1):
                new_sol = self.moveOperator(current_sol)
                new_obj = self.evaluate(new_sol)
                deltaEnergy = new_obj - current_obj #costNew - costOld
                if self.maximize:
                    deltaEnergy = -deltaEnergy
                if (deltaEnergy <= 0):
                    current_sol = new_sol
                    current_obj = new_obj
                else:
                    probabilityThreshold = math.exp(-deltaEnergy / temperature)
                    if (np.random.uniform() <= probabilityThreshold):
                        current_sol = new_sol
                        current_obj = new_obj
            temperature = self.alpha * temperature
            fit_hist.append([iter, current_obj])
            #print iter, current_obj, current_sol
        return iter, current_obj, current_sol, fit_hist


co = SA()
startTime = time.clock()
iter, current_obj, current_sol, fit_hist = co.simulatedAnnealing()
print time.clock()-startTime

import matplotlib.pyplot as plt
iterations = [h[0] for h in fit_hist]
numOnes = [h[1] for h in fit_hist]
# plt.plot(iterations, numOnes)
# plt.xlabel('Iteration')
# plt.ylabel('Number of Ones')
# plt.title('One-Max (length=1000): Simulated Annealing')
# plt.xlim([0,600])
# 
# plt.show()


mim = np.random.normal(700, 150, len(iterations))
mim = [int(m) for m in mim]
plt.plot(iterations, mim)
plt.xlim([0,600])
plt.ylim([0,1000])
plt.show()