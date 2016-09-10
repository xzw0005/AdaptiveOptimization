'''
Created on Sep 10, 2016

@author: XING
'''

import numpy as np
import GeneticAlgorithms.prod.Individual as Individual

class Population(object):
    '''
    classdocs
    '''


    def __init__(self, size, initialize = True):
        '''
        Constructor
        '''
        self.size = size
        population = []
        while size:
            member = Individual.Individual()
            member.randomIndividual()
            self.population.append(member)
            size -= 1
        return population
    
