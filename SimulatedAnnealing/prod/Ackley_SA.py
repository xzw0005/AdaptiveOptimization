'''
Created on Aug 29, 2016
@author: XING WANG    
'''

import numpy as np
import math

def ackley(xx):
    a = 20.0; b = 0.2; c = 2 * math.pi;
    sum1 = 0
    sum2 = 0
    for x in xx:
        sum1 += x**2
        sum2 += math.cos(c * x)
    n = float(len(xx))
    val = -a * math.exp(-b * math.sqrt(sum1 / n)) - math.exp(sum2 / n) + a + math.e
    return val   

class SA(object):
    '''
    Soving a Ackley Function, which has global minimum 0 at [0, 0]
    '''
    
    def __init__(self, current_sol = None, init_temperature=1e3, m=1e3, n=1e3, alpha=.9, lb=-5., ub=5., seed = None):
        '''
        Constructor
        '''
        self.init_temperature = init_temperature
        self.m = m
        self.n = n
        self.alpha = alpha
        self.lb = lb
        self.ub = ub
        if (current_sol is None):                
            self.current_sol = self.getInitSolution(seed)
        else:
            self.current_sol = current_sol
        
    def getInitSolution(self, seed):
        if (seed is not None):
            np.random.seed(seed)
        init_sol = np.random.uniform(size=2)
        lb = self.lb
        ub=self.ub
        init_sol = [lb+xi*(ub-lb) for xi in init_sol]
        return np.array(init_sol)
        
    def moveOperator(self, current_sol, temperature):
        ub = self.ub
        lb = self.lb
        step = min(5e-2, temperature * 1e2/ self.init_temperature)
        u = np.random.random(len(current_sol))
        new_sol = []
        no_dir = 0
        for i in range(len(current_sol)):
            if (u[i] < 1./3):
                direction = -1
            elif (u[i] > 2./3):
                direction = 1
            else:
                direction = 0
                no_dir += 1
            xnew_i = current_sol[i] + direction * step
            if (xnew_i > ub):
                xnew_i = lb + (xnew_i - ub)
            elif (xnew_i < lb):
                xnew_i = ub - (lb - xnew_i)
            new_sol.append(xnew_i)
        if no_dir == len(current_sol):
            return self.moveOperator(current_sol, temperature)
        return np.array(new_sol)
    
    def computeCost(self, x):
        return ackley(x)
    
    def simulatedAnnealing(self):
        current_sol = self.current_sol
        temperature = self.init_temperature 
        alpha=self.alpha
        m=self.m
        n=self.n
        
        current_obj = self.computeCost(current_sol)
        iter = 0 
        while (iter < m) and (current_obj > 1e-10):
            iter += 1
            for i in np.arange(n - 1):
                new_sol = self.moveOperator(current_sol, temperature)
                new_obj = self.computeCost(new_sol)
                deltaEnergy = new_obj - current_obj #costNew - costOld
                if (deltaEnergy <= 0):
                    current_sol = new_sol
                    current_obj = new_obj
                else:
                    probabilityThreshold = math.exp(-deltaEnergy / temperature)
                    if (np.random.uniform() <= probabilityThreshold):
                        current_sol = new_sol
                        current_obj = new_obj
            temperature = self.alpha * temperature
            print iter, current_obj, current_sol
        return iter, current_obj, current_sol


ak = SA()
#print math.log(1e5, 1/.99)
#print ak.init_temperature
print ak.simulatedAnnealing()
