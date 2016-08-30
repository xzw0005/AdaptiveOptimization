'''
Created on Aug 29, 2016

@author: XING WANG    
'''

import numpy as np

class SA(object):
    '''
    Soving a Quadratic Assignment Problem of Department Locations, 
    by minimizing the total transportation through all departments.
    This is Q1 for HW1.
    '''
    

    def __init__(self, X0, T0, m, n, alpha):
        '''
        Constructor
        '''
        self.X0 = X0
        self.T0 = T0
        self.m = m
        self.n = n
        self.alpha = alpha
        
    def moveOperator(self, xOld):
        return None
    
    
    def getCost(self, x):
        return None
    
    def simulatedAnnealing(self):
        xOld = self.X0
        costOld = self.getCost(xOld)
        temperature = self.T0
        m = self.m
        n = self.n
        for time in np.arange(m - 1):
            for i in range(n - 1):
                xNew = self.moveOperator(xOld)
                costNew = self.getCost(xNew)
                deltaEnergy = costNew - costOld
                if (deltaEnergy <= 0):
                    xOld = xNew
                    costOld = costNew
                else:
                    probabilityThreshold = np.exp(-deltaEnergy / temperature)
                    if (np.random.uniform() <= probabilityThreshold):
                        xOld = xNew
                        costOld = costNew
                    
            temperature = self.alpha * temperature
            
        return xOld
    
    