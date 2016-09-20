'''
Created on Sep 9, 2016
@author: Xing Wang
'''
import math
import numpy as np
# import collections
# from timeit import itertools
from EvolutionaryStrategies.prod.AckleyFunction import ackley as ak


class Individual(object):
    
    def __init__(self, gen = 0, xx = None, sigma = None):
        '''
        Constructor
        '''
        self.gen = gen
        if (xx is not None):
            self.xx = xx
            self.fval = ak(xx=xx)
            if (sigma is None):
                self.sigma = np.zeros(len(xx)) + 1
            else:
                self.sigma = sigma
        else:
            self.xx = None
            self.fval = None
            self.sigma = None
        
    def randomIndividual(self, seed = None, bounds = [(-32.0, 32.0)] * 2):
        if seed is not None:
            np.random.seed(seed)
        xx = []
        for i in np.arange(len(bounds)):
            lower = bounds[i][0]
            higher = bounds[i][1]
            x = np.random.random() * (higher - lower) + lower
            xx.append(x)
        #print np.array(xx)
        self.xx = np.array(xx)
        self.fval = self.ackley(xx)
        self.sigma = np.zeros(len(xx)) + 1
        
    def mutate(self):
        bounds = [(-32.0, 32.0)] * 2
        gen = self.gen
        xx = self.xx
        sigma = self.sigma
        #print "original: ", original.xx
        for i in np.arange(len(xx)):
            noise = np.random.normal(0, sigma[i], 1)
            while (bounds[i][0] > xx[i] + noise or bounds[i][1] < xx[i]+noise):
                noise = np.random.normal(0, sigma[i], 1)
            xx[i] = xx[i] + noise
#         noise = np.random.normal(0, sigma, len(xx))
#         xx = xx + noise

        child = Individual(gen, xx, sigma)
        #child.fval = ackley(xx)
        return child
    
    def ackley(self, xx, a = 20.0, b = 0.2, c = 2 * math.pi):
        sum1 = 0
        sum2 = 0
        for x in xx:
            sum1 += x**2
            sum2 += math.cos(c * x)
        n = float(len(xx))
        val = -a * math.exp(-b * math.sqrt(sum1 / n)) - math.exp(sum2 / n) + a + math.e
        return val
    
    def getfval(self):
        self.fval = self.ackley(self.xx)
        has_fval = True
        return self.fval, has_fval

    def __repr__(self):
        return self.xx.__repr__()+ ', %.3f' %self.fval + ',%.3f' %ak(self.xx)+ '\n' #+ " with Ackley Value: " + '%.5f' %self.fval +"~~"+ str(ackley(self.xx))


# class Bounds(object):
#     """Defines a basic bound for numeric lists."""
#     def __init__(self, lower = None, upper = None):
#         self.lower = lower
#         self.upper = upper
#         if self.lower is not None and self.upper is not None:
#             if not isinstance(self.lower, collections.Iterable):
#                 self.lower = itertools.repeat(self.lower)
#             if not isinstance(self.upper, collections.Iterable):
#                 self.upper = itertools.repeat(self.upper)
# 

 
# tryc = Individual([2, 2])
# print tryc.fval
# tryc.randomIndividual()
# print tryc