'''
Created on Sep 20, 2016

@author: XING
'''
import abc
import numpy as np
from EvolutionaryStrategies.prod.AckleyFunction import ackley

class BaseIndividual(object):
    '''
    classdocs
    '''


    def __init__(self, xx = None, chromosome = None):
        '''
        Constructor
        '''
        if chromosome is None:
            self.chromosome = {}
        else:
            self.chromosome = chromosome
        
        if xx is None:
            self.chromosome['xx'] = np.random.uniform(32, -32, 2).astype(dtype='longdouble')
        else:
            self.chromosome['xx'] = xx
            
    @abc.abstractmethod
    def mutate(self):
        pass
    
    @abc.abstractmethod
    def updateIndividualSigma(self, proportion):
        pass
    
    def fitness(self):
        hasFit = False
        if not hasattr(self, 'fitnessValue'):
            self.fitnessValue = ackley(self.chromosome['xx'])
            hasFit = True
        return self.fitnessValue, hasFit
    
class Mutation(BaseIndividual):
    
    def __init__(self, xx=None, sigma):
        self.chromosome = {}
        self.chromosome['sigma'] = sigma
        
        super().__init__(xx, self.chromosome)
       
        
    def updateIndividualSigma(self, proportion):
        self.chromosome['sigma'] = [s*proportion for s in self.chromosome['sigma']]
        pass
    
    def mutate(self):
        #sigma = [s + np.random(0, 1, 1)[0] for s in self.chromosome['sigma']] 
        sigma = self.chromosome['sigma']
        xx = [x + np.random.normal(0, sigma[i], 1)[0] for i, x in enumerate(self.chromosome['xx'])]
        return Mutation(xx, sigma)