'''
Created on Sep 20, 2016

@author: XING
'''
import numpy as np
from EvolutionaryStrategies.prod.AckleyFunction import ackley
import abc
from EvolutionaryStrategies.test.Individual import BaseIndividual as Individual


class EvolutionStrategy(object):
    '''
    classdocs
    '''


    def __init__(self, mu, pressure, sigma, population, maxiter = 50, strategy = "plus", 
                 discrete_dual = False, global_intermediate = False, #intermediate_dual = False, global_discrete = False, 
                 bounds = [(-32.0, 32.0)] * 2, obj = "min"):
        '''
        strategy could be either "plus" or "comma":
        recombination only valid if strategy == "comma"
             could be "discrete_dual", or "intermediate_dual",
              or "global_discrete", or "global_intermediate"
            Only required to implement "discrete_dual" & "global_intermediate", would implement "intermediate_dual" & "global_discrete" later
        obj could be either "min" or "max"
        '''
        self.mu = int(mu)
        self.pressure = pressure
        self.maxiter = maxiter
        self.generation = 0
        if (population == None):
            self.population = self.initPop()
        else:
            self.population = population
        self.obj = obj
        self.goal = 0
        self.tol = 1e-3
        self.bounds = bounds  
        
        self.recombination = "discrete"  
        
    def run(self):
        records = []
        self.succ = 1
        self.fail = 1
        while not self.termination(self.population):
            self.population, result = self.evolution(self.population)
            self.oneFifthRule(self.population)
            records.append(result)
        return records
    
    def termination(self, population):
        return self.generation > self.maxiter or self.bestFitness(population) < self.tol
        
    def oneFifthRule(self, population):
        if (self.avaliation * 1. / self.maxiter).is_integer():
            succ_rate = self.succ * 1. / self.fail
            if succ_rate > 0.2:
                self.updateMu(1/0.85)
            elif succ_rate < 0.2:
                self.updateMu(0.85)
        pass
    
    def updateMu(self, proportion):
        for individual in self.population:
            individual.updateIndividualMu(proportion)
        pass
    
    def getChild(self, population):
        parent = np.random.choice(population, size = 1)
        
    def dualGetChild(self, population):
        parents = np.random.choice(population, size = 2, replace = False)
        child = self.recombinate(parents[0], parents[1])
        return child.mutate()
    
    @abc.abstractmethod
    def bestFitness(self, population):
        pass
    
    @abc.abstractmethod
    def meanFitness(self, population):
        pass
    
    @abc.abstractmethod
    def evolution(self, population):
        pass
    
    def dualRecombination(self, parent1, parent2):
        xx1 = parent1.chromosome['xx']
        xx2 = parent2.chromosome['xx']
        if self.recombination == "discrete":
            inx = np.random.randint(0, 2, size = len(xx1))
            child_xx = []
            for i in range(len(xx1)):
                if inx[i] == 0:
                    child_xx.append(xx1[i])
                else:
                    child_xx.append(xx2[i])
        else: # "intermediate"
            child_xx = [(x+y)/2.0 for x, y in zip(xx1, xx2)]
        return Individual.BaseIndividual(xx=child_xx)
    
